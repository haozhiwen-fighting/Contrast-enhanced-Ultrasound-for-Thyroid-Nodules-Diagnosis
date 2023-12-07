# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        y_test=[]
        y_sc=[]
        y_true=[]
        y_pre=[]
        fae=[]
        for i, batch in enumerate(test_loader):
            if isinstance(batch, list):
                data_us,data_ceus,val_labels, train_mask, tr_mask,clslabel,filename= batch 
                clslabel=np.array(clslabel, dtype=int)
                clslabel=torch.from_numpy(clslabel)
            else:
                val_inputs, val_labels ,clslabel= batch['image'], batch['label'], batch['cls']
                clslabel=np.array(clslabel, dtype=int)
                clslabel=torch.from_numpy(clslabel)
            if torch.cuda.is_available():
                data_us = torch.autograd.Variable(data_us).cuda()
                data_ceus = torch.autograd.Variable(data_ceus).cuda()
                val_labels = torch.autograd.Variable(val_labels).cuda()
                clslabel = torch.autograd.Variable(clslabel).cuda()
            # val_inputs, val_labels = val_inputs.cuda(), val_labels.cuda()      
            # cla_ouput= model(val_inputs,val_labels)
            data=[data_us,data_ceus]
            mmloss,cla_ouput,tar_us,tar_ceus,feature = model(data,clslabel,infer=False) 
            #########将特征信息保存在表格里
            # info_f = feature.view(-1).cpu().numpy()
            info_f = np.array(feature)
            fae.append(info_f)
            
        
            #########
            cla = nn.Softmax(dim=-1)(cla_ouput)
            cla_sc=cla[0,0]
            cla1=torch.max(cla)
            preds = torch.max(cla, dim=-1)[1].float()
            labels_one_hot=torch.zeros(clslabel.shape[0],2).to(device)
            labels_one_hot=labels_one_hot.scatter_(1, clslabel.view(-1,1).long(), 1).to(device)
            _cla_loss=cla_loss(cla,labels_one_hot)
            # seg_loss=se_loss(tar_ceus[:,2,:,:].unsqueeze(dim=1),val_labels)
            y_test.append(clslabel.cpu().numpy())
            # cla1=cla1.cpu().numpy()
            y_sc.append(cla_sc.cpu().numpy())
            y_true.append(clslabel.data.cpu().numpy())
            y_pre.append(preds.cpu().numpy())
            print(' {}/{}'.format( i, len(test_loader)))
            print('病例{}分类标签为：{}  分类结果为：{}'.format(filename,clslabel.cpu().numpy(),preds.cpu().numpy()))

            running_loss += _cla_loss.item() * data_ceus.size(0)
            running_corrects += torch.sum(preds == clslabel.data)
            val_outputs = torch.sigmoid(tar_ceus[:,2,:,:]).cpu().numpy()
            # val_outputs = val_outputs.cpu().numpy()
            val_outputs[ val_outputs > 0.5] = 1
            val_outputs[ val_outputs != 1] = 0
            img=val_outputs[0,:,:]
            # mean_dice=dice(img,val_labels.cpu().numpy())
            mean_dice=dice(torch.from_numpy(img).cuda(),val_labels)
            im = Image.fromarray(np.uint8(img*255))
            # im.show()
            save_path='D:/HHJ/second/CEUS_TMI/cla/result/'
            name="".join(filename)
            pth=save_path+name+'.png'
            im.save(pth)
            val_labels =val_labels.cpu().numpy()
            # mask=val_labels[0,0,:,:]
            
            print(' {}/{}'.format( i, len(test_loader)))
            print("{}Mean Organ Dice: {}".format(filename,mean_dice.cpu().numpy()))
        #     img=np.uint8(img*255)
        #     mask=np.uint8(mask*255)
        #     precisipon=calPrecision(img,mask)
        #     Recall=calRecall(img,mask)
        #     IOU1,DICE=IntersectionOverUnion(mask,img)
            
            dice_list_case.append(mean_dice.cpu().numpy())
        #     recall_list_case.append(Recall)
        #     presion_list_case.append(precisipon)
        #     IOU_list_case.append(IOU1)
        #     DICE_list_case.append(DICE)
        fea_=np.array(fae).reshape(2,124)
        pd.DataFrame(fea_).to_csv('D:/HHJ/second/CEUS_TMI/cla/sample3.csv')
        print("Overall Mean Dice: {}   fangcha{}".format(np.mean(dice_list_case),np.std(dice_list_case)))
        # print("Recall计算结果   Recall    = {}   fang{}".format(np.mean(recall_list_case),np.std(recall_list_case)))
        # print('Precision计算结果 Precision = {}  fang  {}'.format(np.mean(presion_list_case),np.std(presion_list_case)))
        # print('iou计算结果 IOU = {}    fang{}'.format(np.mean(IOU_list_case),np.std(IOU_list_case)))
        # print('iou计算结果 dice = {}   fang{}'.format(np.mean(DICE_list_case),np.std(DICE_list_case)))
        fpr,tpr, thresholds = roc_curve(y_test,y_sc)
        auc =1- roc_auc_score(y_test,y_sc)
        acc = accuracy_score(y_true,y_pre)
        prec = precision_score(y_true,y_pre)
        recall = recall_score(y_true,y_pre)
        f1 = f1_score(y_true,y_pre)

        epoch_loss = running_loss / test_size
        epoch_acc = running_corrects.double() / test_size
        print("classfication loss: {} classfication acc:{}".format(epoch_loss,epoch_acc))
        # plt.plot(fpr,tpr,label='ROC')
        # plt.xlabel('FPR')
        # plt.ylabel('TPR')
        # plt.show()
        plt.plot(fpr,tpr,color='darkorange',label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.savefig('suhan.jpg',dpi=800)

        plt.show()
        print('[ACC,AUC,Pre,Recall,f1]=[{},{},{},{},{}]'.format(acc,auc,prec,recall,f1))
        print("threshold:{}".format(thresholds))
if __name__ == '__main__':
    main()