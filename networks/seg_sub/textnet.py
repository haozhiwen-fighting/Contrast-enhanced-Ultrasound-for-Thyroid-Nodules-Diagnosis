import torch
import torch.nn as nn
import torch.nn.functional as F
from networks.seg_sub.vgg import VggNet
from networks.seg_sub.resnet import ResNet
# from networks.seg_sub.vnetc import VNet
from networks.seg_sub.Vnet_exchange import VNet
from util.config import config as cfg
import time

class UpBlok(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, upsampled, shortcut):
        x = torch.cat([upsampled, shortcut], dim=1)
        x = self.conv1x1(x)
        x = F.relu(x)
        x = self.conv3x3(x)
        x = F.relu(x)
        x = self.deconv(x)
        return x


class RRGN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.FNUM = len(cfg.fuc_k)
        self.SepareConv0 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0),
        )
        channels2 = in_channels + 1
        self.SepareConv1 = nn.Sequential(
            nn.Conv2d(channels2, channels2, kernel_size=(5, 1), stride=1, padding=1),
            nn.Conv2d(channels2, channels2, kernel_size=(1, 5), stride=1, padding=1),
            nn.Conv2d(channels2, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        f_map = list()
        for i in range(self.FNUM):
            if i == 0:
                f = self.SepareConv0(x); f_map.append(f); continue
            b1 = torch.cat([x, f_map[i - 1]], dim=1)
            f = self.SepareConv1(b1)
            f_map.append(f)
        f_map = torch.cat(f_map,dim=1)
        return f_map


class FPN(nn.Module):

    def __init__(self, backbone='vgg_bn', pre_train=True):
        super().__init__()
        self.backbone_name = backbone

        if backbone == "vgg" or backbone == 'vgg_bn':
            if backbone == 'vgg_bn':
                self.backbone = VggNet(name="vgg16_bn", pretrain=pre_train)
            elif backbone == 'vgg':
                self.backbone = VggNet(name="vgg16", pretrain=pre_train)

            self.deconv5 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(512 + 256, 128)
            self.merge3 = UpBlok(256 + 128, 64)
            self.merge2 = UpBlok(128 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)

        elif backbone == 'resnet50' or backbone == 'resnet101':
            if backbone == 'resnet101':
                self.backbone = ResNet(name="resnet101", pretrain=pre_train)
            elif backbone == 'resnet50':
                self.backbone = ResNet(name="resnet50", pretrain=pre_train)

            self.deconv5 = nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1)
            self.merge4 = UpBlok(1024 + 256, 128)
            self.merge3 = UpBlok(512 + 128, 64)
            self.merge2 = UpBlok(256 + 64, 32)
            self.merge1 = UpBlok(64 + 32, 16)
        elif backbone == 'vnet':
            self.backbone = VNet(elu=True, nll=False)
            model_path='D:/HHJ/second/CEUS_TMI/cla/pre/model.pt'
            print("load the weight from {}".format(model_path))
            checkpoint=torch.load(model_path) 
            self.backbone.load_state_dict(checkpoint['state_dict'])
        else:
            print("backbone is not support !")

    def forward(self, x):
        up1 = self.backbone(x)
        return up1


class TextNet(nn.Module):

    def __init__(self, backbone='vnet', is_training=True):
        super().__init__()
        self.is_training = is_training 
        self.backbone_name = backbone
        self.fpn = FPN(self.backbone_name, pre_train=is_training)
        self.rrgn = RRGN(2)


    def forward(self, x):
        # x=x.squeeze(dim=1)
        end = time.time()
        up1,x_tw,out64 = self.fpn(x)
        b_time = time.time()-end
        end = time.time()
        predict_out0=up1[0]
        predict_out = self.rrgn(up1[1])
        # predict_out = self.rrgn(up1)
        iter_time = time.time()-end
        
        return predict_out0,predict_out,x_tw,out64, b_time, iter_time
