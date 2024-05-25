
import os
import numpy as np
import torch
import random
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from PIL import Image
from torch import nn
from networks.cla_3D_family.C3D_model import C3D
from networks.cla_3D_family.R3D_model import R3DClassifier
from networks.cla_3D_family.R2plus1D import R2Plus1DClassifier
from networks.key_c3d import C3D_mu
# from networks.cla_pro import cla_p
from utils.us_ceusdata import USdata
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from functools import partial
import argparse
from trainer import dice

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--checkpoint', default=None, help='start training from saved checkpoint')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
# parser.add_argument('--data_dir', default='', type=str, help='dataset directory')
parser.add_argument('--data_dir', default='', type=str, help='dataset directory')
parser.add_argument('--pretrained_model_name', default='model.pt', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=100, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=4, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=1, type=int, help='number of sliding window batch size')  #2
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')     #1e-4
parser.add_argument('--optim_name', default='adam', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-5, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')  #这个变量有何用？
parser.add_argument('--val_every', default=1, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')   #多GPU分布式训练
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=1, type=int, help='number of workers')
parser.add_argument('--model_name', default='unetr', type=str, help='model name')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')   #16
parser.add_argument('--in_channels', default=3, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=3, type=int, help='number of output channels')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=50, type=int, help='number of warmup epochs')     #预热步数
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--resume_jit', action='store_true', help='resume training from pretrained torchscript checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')
parser.add_argument('--classification', default=False, type=bool, help='constant added to dice denominator to avoid nan')
parser.add_argument('--frames', default=7, type=int, help='the len of video')
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from util.config import config as cfg
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        inputtarget=input_tensor.detach().cpu().numpy()
        # a=inputtarget[0]
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target=target != 0 
        target=target.float()
        # target = self._one_hot_encoder(target).squeeze(dim=2)
        # if weight is None:
        #     weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        # diceloss=[]
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            # loss += dice * weight[i]
            loss += dice 
        return loss / self.n_classes

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    args.logdir = './runs/' + args.logdir
    setup_seed(20)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method('fork', force=True)
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend,
                                init_method=args.dist_url,
                                world_size=args.world_size,
                                rank=args.rank)
    torch.cuda.set_device(args.gpu)
    # torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.benchmark = False
    args.test_mode = False

    #数据加载
    train_set = USdata(args.data_dir, preprocess=True, split = 'train',frames=16) 
    val_set = USdata( args.data_dir, preprocess=True,split = 'val' ,frames=16)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=1, persistent_workers=True,drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, num_workers=1, persistent_workers=True)
    print(args.rank, ' gpu', args.gpu)
    if args.rank == 0:
        print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)  
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if (args.model_name is None) or args.model_name == 'unetr':     #模型实例化       
        # model=classfication(in_channels=3,out_channels=3)
        model=C3D(num_classes=2)
        # model=R3DClassifier(num_classes=2)
        # model=R2Plus1DClassifier(num_classes=2)
        # model=cla_p()
        # model_path=''
        # checkpoint=torch.load(model_path) 
        # model.load_state_dict(checkpoint['state_dict'])
        # model=C3D_mu(2)
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))
    
    # dice_loss = DiceLoss(to_onehot_y=True,
    #                     softmax=True,
    #                     squared_pred=True,
    #                     smooth_nr=args.smooth_nr,
    #                     smooth_dr=args.smooth_dr)
    cla_loss=nn.CrossEntropyLoss()
    criterion = TextLoss()
    loss_fnc=AutomaticWeightedLoss(criterion,cla_loss,2).cuda()
    seg_loss = DiceLoss(1)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == 'batch':
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.gpu],
                                                          output_device=args.gpu,
                                                          find_unused_parameters=True)
    if args.optim_name == 'adam':
        # optimizer = torch.optim.Adam(model.parameters(),
        #                              lr=args.optim_lr,
        #                              weight_decay=args.reg_weight)
        optimizer = torch.optim.Adam([{"params":model.parameters()},{"params":loss_fnc.parameters()}],
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrschedule == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrschedule == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    accuracy = run_training(
                            model=model,
                            train_loader=train_loader,
                            val_loader=val_loader,
                            optimizer=optimizer,
                            loss_func=loss_fnc,
                            cla_loss=cla_loss,
                            seg_loss=seg_loss,
                            args=args,
                            scheduler=scheduler,
                            start_epoch=start_epoch,
                             )
    return accuracy

if __name__ == '__main__':
    main()
