import os
import timeit
from datetime import datetime
import time
import shutil
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from torch.cuda.amp import GradScaler, autocast
from tensorboardX import SummaryWriter
import torch.nn.parallel
import torch.utils.data.distributed
from torchvision import transforms
import matplotlib.pyplot as plt
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
        score = torch.sigmoid(score)
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
        target = self._one_hot_encoder(target).squeeze(dim=2)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        # diceloss=[]
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes
class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params,requires_grad=True)

    def forward(self, *x):
        loss_sum = 0
        # b=self.params[1]
        # a=self.params[0]
        for i, loss in enumerate(x):
            loss1=0
            loss1= 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
            loss_sum=loss_sum+loss1
        return loss_sum
class BCEDiceLoss11(nn.Module):
    def __init__(self):
        super().__init__()
    def one_hot_encoder(self, input_tensor):
        tensor_list = []
        # inputtarget=input_tensor.detach().cpu().numpy()
        # a=inputtarget[0]
        for i in range(2):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def forward(self, input, target):
        #####one-shot
        # target=self.one_hot_encoder(target)
        # bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        # return 0.5 * bce + dice
        return dice
class Task_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.sig_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.sig_weight.data.fill_(0.25)    
    def forward(self,p_con,p_pre):
        a=p_con-p_pre
        loss=np.linalg.norm(a,ord=2)
        loss=loss*loss
        loss=loss/(self.sig_weight*self.sig_weight*2+0.000001)+np.log(self.sig_weight)
        return loss
def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)


def train_epoch(model,
                train_loader,
                classfic,
                optimizer,
                scaler,
                epoch,
                loss_func,
                cla_loss,
                seg_loss,
                args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    clss_loss = AverageMeter()
    # task_loss=Task_Loss()
    aw1 = AutomaticWeightedLoss(2).cuda()
    train_size = len(train_loader)
    print(train_size)
    for idx, batch_data in enumerate(tqdm(train_loader)):   ## bie gai     
        if isinstance(batch_data, list):
            data_us,data_ceus, target,train_mask, tr_mask,clslabel ,filename= batch_data
            clslabel=np.array(clslabel, dtype=int)
            clslabel=torch.from_numpy(clslabel)
        else:
             data_us,data_ceus, target,clslabel,fileneame = batch_data['image_us'],batch_data['image'],  batch_data['label'], batch_data['cls'],batch_data['filename']
             clslabel=np.array(clslabel, dtype=int)
             clslabel=torch.from_numpy(clslabel)
        data_us,data_ceus, target = data_us.cuda(args.rank),data_ceus.cuda(args.rank), target.cuda(args.rank)
        if torch.cuda.is_available():
            data_us = torch.autograd.Variable(data_us).cuda()
            data_ceus = torch.autograd.Variable(data_ceus).cuda()
            target = torch.autograd.Variable(target).cuda()
            clslabel = torch.autograd.Variable(clslabel).cuda()
            train_mask = torch.autograd.Variable(train_mask).cuda()
            tr_mask = torch.autograd.Variable(tr_mask).cuda()
        for param in model.parameters(): param.grad = None
        with autocast(enabled=args.amp):
            labels_one_hot=torch.zeros(clslabel.shape[0],2).to(device)
            labels_one_hot=labels_one_hot.scatter_(1, clslabel.view(-1,1).long(), 1).to(device)
            data=[data_us,data_ceus]
            mmloss,cla,tar_us,tar_ceus = model(data,clslabel,infer=False)          
            cla = nn.Softmax(dim=-1)(cla)
            # _cla_loss=mmloss+cla_loss(cla,labels_one_hot)
            # seg_loss=loss_func(tar_us,tar_ceus,train_mask, tr_mask,target)
            _cla_loss,seg_loss,cla_loss,par1,par2,par3=loss_func(tar_us,tar_ceus,train_mask, tr_mask,target,cla,labels_one_hot,mmloss)
            # tar_loss=task_loss()
            # _cla_loss_=cla_loss(cla,labels_one_hot)
            # # loss = loss_func(output, train_mask, tr_mask)
            # _cla_loss=_cla_loss+seg_loss
            # _cla_loss = aw1(_cla_loss, seg_loss)
            # _cla_loss=_cla_loss_
        if args.amp:
            scaler.scale(_cla_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else: 
            _cla_loss.backward()
            optimizer.step()
        
        clss_loss.update(_cla_loss.item(), n=args.batch_size)
        start_time = time.time()
    for param in model.parameters() : param.grad = None

    return clss_loss.avg,seg_loss.detach().cpu().numpy(),cla_loss.detach().cpu().numpy(),par1.detach().cpu().numpy(),par2.detach().cpu().numpy(),par3
   

def val_epoch(model,
              val_loader,
              classfic,
              epoch,
              args,
              cla_loss,):
    model.eval()
    start_time = time.time()
    val_loss = AverageMeter()
    val_loss1 = AverageMeter()
    se_loss=BCEDiceLoss11()
    cl_loss=nn.CrossEntropyLoss()
    running_corrects=0.0
    runing_false=0.0
    val_size=len(val_loader)*args.batch_size
    with torch.no_grad():
        for idx, batch_data in tqdm(enumerate(val_loader)):
            if isinstance(batch_data, list):
                # data, target,train_mask,tr_mask,clslabel ,filename= batch_data
                data_us,data_ceus, target,train_mask, tr_mask,clslabel ,filename= batch_data
                clslabel=np.array(clslabel, dtype=int)
                clslabel=torch.from_numpy(clslabel)
            else:
                # data, target,train_mask,tr_mask,clslabel,fileneame = batch_data['image'],  batch_data['label'],  batch_data['train_mask'],  batch_data['tr_mask'],batch_data['cls'],batch_data['filename']
                data_us,data_ceus, target,clslabel,fileneame = batch_data['image_us'],batch_data['image'],  batch_data['label'],  batch_data['cls'],batch_data['filename']
                clslabel=np.array(clslabel, dtype=int)
                clslabel=torch.from_numpy(clslabel)
            data_us,data_ceus, target = data_us.cuda(args.rank), data_ceus.cuda(args.rank),target.cuda(args.rank)
            # train_mask,tr_mask=train_mask.cuda(args.rank),tr_mask.cuda(args.rank)
            if torch.cuda.is_available():
                data_us = torch.autograd.Variable(data_us).cuda()
                data_ceus = torch.autograd.Variable(data_ceus).cuda()
                target = torch.autograd.Variable(target).cuda()
                clslabel = torch.autograd.Variable(clslabel).cuda()
                train_mask = torch.autograd.Variable(train_mask).cuda()
                tr_mask = torch.autograd.Variable(tr_mask).cuda()
            for param in model.parameters(): param.grad = None
            data=[data_us,data_ceus]
            _cla_loss_,cla,tar_us,tar_ceus = model(data,clslabel,infer=True)
            
            cla = nn.Softmax(dim=-1)(cla)
            preds = torch.max(cla, dim=-1)[1].float()
            labels_one_hot=torch.zeros(clslabel.shape[0],2).to(device)
            labels_one_hot=labels_one_hot.scatter_(1, clslabel.view(-1,1).long(), 1).to(device)
            seg_loss=se_loss(tar_ceus[:,2,:,:].unsqueeze(dim=1),target)
            _cla_loss=cl_loss(cla,labels_one_hot)
            # _cla_loss=_cla_loss_
            val_loss1.update(_cla_loss.item(), n=args.batch_size)  
            #########
            running_corrects += torch.sum(preds == clslabel.float().data)
            runing_false += torch.sum(preds != clslabel.float().data)
            # print('此病例{}的标签为：{}'.format(filename,clslabel))
            # print('此病例{}的分类结果为：{}'.format(filename,preds))
            start_time = time.time()
        epoch_acc = running_corrects.double() / (running_corrects+runing_false)

   
    return val_loss1.avg,epoch_acc,seg_loss
    

def save_checkpoint(model,
                    epoch,
                    args,
                    fold_idx=0,
                    filename='model_cla.pt',
                    best_acc=0,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': epoch,
            'best_acc': best_acc,
            'fold':fold_idx,
            'state_dict': state_dict
            }
    if optimizer is not None:
        save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print('Saving checkpoint', filename)

def run_training(
                model,
                #  dataset,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 cla_loss,
                 seg_loss,
                #  acc_func,
                 args,
                 scheduler=None,
                 start_epoch=0,
                 ):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    class_max_acc=0.0
    best_epoch=0
    classfic=0
    #######
    x = []
    train_loss_list = []
    val_loss_list = []
    val_acc=[]
    bestacc=[]
    ########
    xzhou=[]
    class_max_acc=0.0
    class_min_loss=10000.0
    for epoch_idx in range(args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch_idx)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), 'Epoch:', epoch_idx)
        epoch_time = time.time()
        clas_loss,seg_loss,cla_loss,par1,par2,par3 = train_epoch(model,
                            train_loader,
                            classfic,
                            optimizer,
                            scaler=scaler,
                            epoch=epoch_idx,
                            loss_func=loss_func,
                            cla_loss=cla_loss,
                            seg_loss=seg_loss,
                            args=args)
    
        print('Final training  {}/{}'.format(epoch_idx+1, args.max_epochs), 
                'total loss: {:.4f}'.format(clas_loss),
                'seg loss: {:.4f}'.format(seg_loss[0]),
                'classfication loss: {:.4f}'.format(cla_loss),
                'parameters1: {:.4f}'.format(par1),
                'parameters2: {:.4f}'.format(par2),
                'parameters3: {:.4f}'.format(par3),
                'optimizer lr: {}'.format(scheduler.get_last_lr()),
            'time {:.2f}s'.format(time.time() - epoch_time))
    
        if args.rank==0 and writer is not None:
            writer.add_scalar('train_loss', clas_loss, epoch_idx)
        b_new_best = False
        if (epoch_idx+1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            cla_avg_acc,epoch_acc,seg_loss = val_epoch(model,
                                val_loader,
                                classfic,
                                epoch=epoch_idx,
                                args=args,
                                cla_loss=cla_loss,
                                )
            print('Final validation  {}/{}'.format(epoch_idx+1, args.max_epochs ),
                'classfication loss:{:.4f}'.format(cla_avg_acc), 
                'classfication acc:{:.4f}'.format(epoch_acc),
                'segmentation dice:{:.4f}'.format(seg_loss),
                'time {:.2f}s'.format(time.time() - epoch_time))
            model_name='model_finalcla.pt'
            model_name1='model_cla.pt'
            model_name2='model_minlosscla.pt'
            if epoch_acc >= class_max_acc:
                print('new best ({:.6f} --> {:.6f}). '.format(class_max_acc, epoch_acc))
                class_max_acc = epoch_acc
                b_new_best = True
                best_epoch=epoch_idx
                if args.rank == 0 and args.logdir is not None :
                    save_checkpoint(model, epoch_idx, args,
                                    # old_idx=fold_idx,
                                    best_acc=class_max_acc,
                                    filename=model_name1,
                                    optimizer=optimizer,
                                    scheduler=scheduler)
            print('the bestepoch is {}/{},the max cla acc:{:.6f} .'.format(best_epoch,args.max_epochs,class_max_acc))
            if cla_avg_acc <= class_min_loss:
                print('new min loss ({:.6f} --> {:.6f}). '.format(class_min_loss, cla_avg_acc))
                class_min_loss = cla_avg_acc
                b_new_best = True
                best_epoch=epoch_idx
                if args.rank == 0 and args.logdir is not None :
                    save_checkpoint(model, epoch_idx, args,
                                    # old_idx=fold_idx,
                                    best_acc=class_max_acc,
                                    filename=model_name2,
                                    optimizer=optimizer,
                                    scheduler=scheduler)
            if args.rank == 0 and args.logdir is not None :
                # model_name='model_finalcla{}'.format(fold_idx)
                save_checkpoint(model,
                            epoch_idx,
                            args,
                            # fold_idx=fold_idx,
                            best_acc=class_max_acc,
                            filename=model_name)
                if b_new_best:
                    print('Copying to model.pt new best model!!!!')
                    shutil.copyfile(os.path.join(args.logdir, model_name), os.path.join(args.logdir, model_name1))

        # if scheduler is not None:
        if epoch_idx<=51:
            scheduler.step()
        xzhou.append(epoch_idx)
        train_loss_list.append(clas_loss)
        val_loss_list.append(cla_avg_acc)
    bestacc.append(class_max_acc)
    print('best acc',bestacc)
    print('cv {}'.format(class_max_acc))
# 绘制loss曲线
    plt.subplot(1,1,1)
    try:
        train_loss_lines1.remove(train_loss_lines1[0])  # 移除上一步曲线
        val_loss_lines1.remove(val_loss_lines1[0])
    except Exception:
        pass

    train_loss_lines1 = plt.plot(xzhou, train_loss_list, 'r', lw=1)  # lw为曲线宽度
    val_loss_lines1 = plt.plot(xzhou, val_loss_list, 'b', lw=1)
    plt.title("loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train_loss","val_acc"])
    plt.show()
    plt.pause(0.1)
    # acc=class_max_acc
    print('Training Finished !, Best Accuracy: ',class_max_acc)

    return class_max_acc