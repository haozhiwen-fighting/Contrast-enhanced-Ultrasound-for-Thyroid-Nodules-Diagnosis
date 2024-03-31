

import os
import torch
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from torch.utils.data import DataLoader
# from utils.vdata import USdata
# from utils.nousdata import USdata
from utils.us_ceusdata import USdata
# from networks.unetr import UNETR
# from networks.unetr import classfication
# from trainer import dice
import argparse
from networks.cla_3D_family.C3D_model import C3D
from networks.cla_3D_family.R3D_model import R3DClassifier
from networks.cla_3D_family.R2plus1D import R2Plus1DClassifier
from networks.cla_pro import cla_p
from torch import nn
import pandas as pd 
import cv2
from sklearn.metrics import roc_curve,roc_auc_score,accuracy_score,precision_score,recall_score, f1_score

parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--pretrained_dir', default='./pre_trained/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='D:/data/CEUS/thyoid video/data_aug1/', type=str, help='dataset directory')
parser.add_argument('--pretrained_model_name', default='model_minlosscla.pt', type=str, help='pretrained model name')   #[model_cla,model_finalcla,model_minlosscla]
parser.add_argument('--saved_checkpoint', default='ckpt', type=str, help='Supports torchscript or ckpt pretrained checkpoint type')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--in_channels', default=3, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--num_heads', default=4, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--workers', default=1, type=int, help='number of workers')
parser.add_argument('--RandFlipd_prob', default=0.2, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.2, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.1, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.1, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
# def dice(x, y):
#     intersect = np.sum(np.sum(np.sum(x * y)))
#     y_sum = np.sum(np.sum(np.sum(y)))
#     if y_sum == 0:
#         return 0.0
#     x_sum = np.sum(np.sum(np.sum(x)))
#     return 2 * intersect / (x_sum + y_sum)
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
        # input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        # dice = 1 - dice.sum() / num
        # return 0.5 * bce + dice
        return dice
def main():
    args = parser.parse_args()
    args.test_mode = True
    test_set = USdata( args.data_dir, preprocess=True,split = 'val' ,frames=16)
    test_loader = DataLoader(test_set, batch_size=1, num_workers=1, persistent_workers=True)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    cla_loss=nn.CrossEntropyLoss()
    if args.saved_checkpoint == 'torchscript':
        model = torch.load(pretrained_pth)
    elif args.saved_checkpoint == 'ckpt':
        # model=classfication(in_channels=3,out_channels=3)
        model=cla_p()
        # model=C3D(2)
        # model=R3DClassifier(2)
    checkpoint=torch.load(pretrained_pth)        
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)
    test_size = len(test_loader)
    print(test_size)
    running_corrects=0.0
    running_loss=0.0
    dice_list_case=[]
    dice=BCEDiceLoss11()
    with torch.no_grad():
     
        print('[ACC,AUC,Pre,Recall,f1]=[{},{},{},{},{}]'.format(acc,auc,prec,recall,f1))
        print("threshold:{}".format(thresholds))
if __name__ == '__main__':
    main()
