import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch.nn.functional as F
# class C3D_attention(nn.Module):

#     def __init__(self, num_classes):
#         super(C3D, self).__init__()

#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         self.batchnorm1 = nn.BatchNorm3d(64)

#         self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
#         self.batchnorm2 = nn.BatchNorm3d(128)

#         self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
#         self.batchnorm3 = nn.BatchNorm3d(256)
#         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
#         self.batchnorm4 = nn.BatchNorm3d(512)
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.fc1 = nn.Linear(13824, 512)
        
#         self.fc2 = nn.Linear(512, num_classes)

#         self.dropout = nn.Dropout(p=0.5)

#         self.relu = nn.ReLU()

#         # SPP
#         self.spp = SPP()


#     def forward(self, x,x1):
#         x=x.permute(0,2,1,3,4)
#         x1=x1.repeat(32,1,1,1,1).permute(1,2,0,3,4)
#         x=x*x1

#         # batch_size,channels,帧数,H,W
#         x = self.relu(self.batchnorm1(self.conv1(x)))
#         x = self.relu(self.batchnorm2(self.conv2(x)))
#         x = self.relu(self.batchnorm3(self.conv3(x)))
#         x = self.dropout(x)
#         x = self.pool1(x)

#         x = self.relu(self.batchnorm4(self.conv4(x)))
#         # x = self.dropout(x)
#         x = self.pool2(x)

#         x = self.spp(x)
    
#         x = x.view(-1, 13824)
#         x = self.relu(self.fc1(x))
#         x = self.dropout(x)
#         outputs1 = self.fc2(x)

#         return outputs1
# class C3D(nn.Module):

#     def __init__(self, num_classes):
#         super(C3D, self).__init__()

#         self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         # self.conv1 = nn.Conv3d(8, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
#         self.batchnorm1 = nn.BatchNorm3d(64)

#         self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
#         self.batchnorm2 = nn.BatchNorm3d(128)

#         self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
#         self.batchnorm3 = nn.BatchNorm3d(256)
#         self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.conv4 = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
#         self.batchnorm4 = nn.BatchNorm3d(512)
#         self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

#         self.fc1 = nn.Linear(25088, 512)
        
#         # self.fc2 = nn.Linear(512, num_classes)
#         self.fc2 = nn.Linear(1536, num_classes)
#         self.dropout = nn.Dropout(p=0.5)

#         self.relu = nn.ReLU()


#         # Attention
#         self.attention_conv1 = nn.Conv3d(256, 64, kernel_size=(1, 3, 3), stride=(1, 2, 2))

#         self.attention_conv2 = nn.Conv3d(64, 16, kernel_size=(1, 3, 3), stride=(1, 2, 2))

#         self.attention_conv3 = nn.Conv3d(16, 1, kernel_size=(1, 2, 2), stride=(1, 1, 1))

#         # SPP
#         self.spp = SPP()


#     def forward(self, x,x1,x2,x3):
#         # x=x.permute(0,2,1,3,4)
#         x1=x1.repeat(16,1,1,1,1).permute(1,2,0,3,4)
#         x2=x2.repeat(16,1,1,1,1).permute(1,2,0,3,4)
#         x3=x3.repeat(16,1,1,1,1).permute(1,2,0,3,4)
#         x1=x*x1
#         x2=x*x2
#         x3=x*x3

#         # batch_size,channels,帧数,H,W
#         x1 = self.relu(self.batchnorm1(self.conv1(x1)))
#         x1 = self.relu(self.batchnorm2(self.conv2(x1)))
#         x1 = self.relu(self.batchnorm3(self.conv3(x1)))
#         x1 = self.dropout(x1)
#         x1 = self.pool1(x1)
#         x1 = self.relu(self.batchnorm4(self.conv4(x1)))
#         x1 = self.pool2(x1)
#         x1 = x1.view(-1, 25088)
#         x1 = self.relu(self.fc1(x1))
#         x1 = self.dropout(x1)

#         x2 = self.relu(self.batchnorm1(self.conv1(x2)))
#         x2 = self.relu(self.batchnorm2(self.conv2(x2)))
#         x2 = self.relu(self.batchnorm3(self.conv3(x2)))
#         x2 = self.dropout(x2)
#         x2 = self.pool1(x2)
#         x2 = self.relu(self.batchnorm4(self.conv4(x2)))
#         x2 = self.pool2(x2)
#         x2 = x2.view(-1, 25088)
#         x2 = self.relu(self.fc1(x2))
#         x2 = self.dropout(x2)

#         x3 = self.relu(self.batchnorm1(self.conv1(x3)))
#         x3 = self.relu(self.batchnorm2(self.conv2(x3)))
#         x3 = self.relu(self.batchnorm3(self.conv3(x3)))
#         x3 = self.dropout(x3)
#         x3 = self.pool1(x3)
#         x3 = self.relu(self.batchnorm4(self.conv4(x3)))
#         x3 = self.pool2(x3)
#         x3 = x3.view(-1, 25088)
#         x3 = self.relu(self.fc1(x3))
#         x3 = self.dropout(x3)
#         outputs = torch.cat([x1, x2, x3], dim=1)
#         outputs1 = self.fc2(outputs)

#         return outputs1

class BasicBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=[1,1],padding=1) -> None:
        super(BasicBlock, self).__init__()
        # 残差部分
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=stride[0],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True), # 原地替换 节省内存开销
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride[1],padding=padding,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut 部分
        # 由于存在维度不一致的情况 所以分情况
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                # 卷积核为1 进行升降维
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride[0], bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.layer(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# 采用bn的网络中，卷积层的输出并不加偏置
class ResNet18(nn.Module):
    def __init__(self, num_classes=2) -> None:
        super(ResNet18, self).__init__()
        self.in_channels = 32
        # 第一层作为单独的 因为没有残差快
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # conv2_x
        self.conv2 = self._make_layer(BasicBlock,32,[[2,1],[1,1]])
        # conv3_x
        self.conv3 = self._make_layer(BasicBlock,64,[[2,1],[1,1]])
        # conv4_x
        self.conv4 = self._make_layer(BasicBlock,128,[[2,1],[1,1]])
        # conv5_x
        self.conv5 = self._make_layer(BasicBlock,256,[[2,1],[1,1]])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(p=0.1)
    def _make_layer(self, block, out_channels, strides):
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    def forward(self, x,x1):
        x=x[:,:,6,:,:]
        x=x*x1
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        # out = self.avgpool(out)
        out =self.dropout(out)
        out = out.reshape(x.shape[0], -1)
        out = self.fc(out)
        out =self.dropout(out)
        return out

class C3D_mu(nn.Module):

    def __init__(self, num_classes):
        super(C3D_mu, self).__init__()
        chnnel=[16,32,64,128]
        self.conv1 = nn.Conv3d(3, chnnel[0], kernel_size=(5, 5, 5), stride=(1, 2, 2), padding=(2, 2, 2))
        # self.conv1 = nn.Conv3d(8, 64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.batchnorm1 = nn.BatchNorm3d(chnnel[0])

        self.conv2 = nn.Conv3d(chnnel[0],chnnel[1], kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
        self.batchnorm2 = nn.BatchNorm3d(chnnel[1])

        self.conv3 = nn.Conv3d(chnnel[1],chnnel[2], kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.batchnorm3 = nn.BatchNorm3d(chnnel[2])
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = nn.Conv3d(chnnel[2], chnnel[3], kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.batchnorm4 = nn.BatchNorm3d(chnnel[3])
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.fc1 = nn.Linear(2048, 512)
        # self.fc = nn.Conv1d(128, 1,1)
        # self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(p=0.1)

        self.relu = nn.ReLU()
    def forward(self, x,x1,x2,tw):
        tw1=tw.sum(dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4).permute(0,2,1,3,4)
        # tw=tw.sum(dim=1).unsqueeze(2).unsqueeze(3).unsqueeze(4).permute(0,2,1,3,4)
        x1=x1.unsqueeze(dim=1)
        x1=x*x1

        # batch_size,channels,帧数,H,W
        # x1 = self.relu(self.batchnorm1(self.conv1(x1)))
        # x1 = self.relu(self.batchnorm2(self.conv2(x1)))
        x1 = self.batchnorm1(self.conv1(x1))
        x1 = self.batchnorm2(self.conv2(x1))
        x1=x1*x2
        x1=x1*tw1
        x1 = self.batchnorm3(self.conv3(x1))
        x1 = self.dropout(x1)
        x1 = self.pool1(x1)
        x1 = self.batchnorm4(self.conv4(x1))
        x1 = self.dropout(x1)
        x1 = self.pool2(x1)
        # x1=tw.unsqueeze(dim=3).unsqueeze(dim=4)*x1
       
        # outputs1 = x1.view(-1, 512)
        outputs1 = x1.view(-1, 2048)
        outputs1 = self.relu(self.fc1(outputs1))
        outputs1 = self.dropout(outputs1)

        return outputs1
