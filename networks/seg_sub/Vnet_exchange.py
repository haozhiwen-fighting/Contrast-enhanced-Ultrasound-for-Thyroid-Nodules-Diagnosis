import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat
# from networks.Vnet.chnnel_exahange import *
import torch
import torch.nn as nn

def conv3x3(in_planes, out_planes, stride=1, bias=False):
    "3x3 convolution with padding"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=3,
                                    stride=stride, padding=1, bias=bias))


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    "1x1 convolution"
    return ModuleParallel(nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                    stride=stride, padding=0, bias=bias))

class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]


class ModuleParallel(nn.Module):
    def __init__(self, module):
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel):
        return [self.module(x) for x in x_parallel]


class BatchNorm3dParallel(nn.Module):
    def __init__(self, num_features, num_parallel):
        super(BatchNorm3dParallel, self).__init__()
        for i in range(num_parallel):
            setattr(self, 'bn_' + str(i), nn.BatchNorm3d(num_features))

    def forward(self, x_parallel):
        return [getattr(self, 'bn_' + str(i))(x) for i, x in enumerate(x_parallel)]

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, num_parallel, bn_threshold, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = BatchNorm3dParallel(planes, num_parallel)
        self.conv2 = conv3x3(planes, planes, stride=stride)
        self.bn2 = BatchNorm3dParallel(planes, num_parallel)
        self.conv3 = conv1x1(planes, planes * 4)
        self.bn3 = BatchNorm3dParallel(planes * 4, num_parallel)
        self.relu = ModuleParallel(nn.ReLU(inplace=True))
        self.num_parallel = num_parallel
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange()
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm3d):
                self.bn2_list.append(module)

    def forward(self, x):
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = [out[l] + residual[l] for l in range(self.num_parallel)]
        out = self.relu(out)

        return out
def passthrough(x, **kwargs):
    return x

def ELUCons(elu, nchan):
    if elu:
        return nn.ELU(inplace=True)
    else:
        return nn.PReLU(nchan)

# normalization between sub-volumes is necessary
# for good performance
class ContBatchNorm3d(nn.modules.batchnorm._BatchNorm): 
    def _check_input_dim(self, input):
        super(ContBatchNorm3d, self).__init__()
        if input.dim() != 5:
            raise ValueError('expected 5D input (got {}D input)'
                             .format(input.dim()))
        # super(ContBatchNorm3d, self)._check_input_dim(input)

    def forward(self, input):
        # self._check_input_dim(input)
        return F.batch_norm(
            input, self.running_mean, self.running_var, self.weight, self.bias,
            True, self.momentum, self.eps)


class LUConv(nn.Module):
    def __init__(self, nchan, elu):
        super(LUConv, self).__init__()
        self.relu1 = ELUCons(elu, nchan)
        self.conv1 = nn.Conv3d(nchan, nchan, kernel_size=5, padding=2)
        self.bn1 = ContBatchNorm3d(nchan)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        return out


def _make_nConv(nchan, depth, elu):
    layers = []
    for _ in range(depth):
        layers.append(ModuleParallel(LUConv(nchan, elu)))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):
    def __init__(self, outChans, elu):
        super(InputTransition, self).__init__()
        self.conv1 =ModuleParallel( nn.Conv3d(1, 8, kernel_size=5, padding=2))
        self.bn1 =BatchNorm3dParallel(8,2)
        self.relu1 =ModuleParallel( ELUCons(elu, 16))
    def forward(self,x):
        # do we want a PRELU here as well?
        out = self.bn1(self.conv1(x))
        # out = self.bn1(x)
        # split input in to 16 channels
        x_us_16 = torch.cat((x[0], x[0], x[0], x[0], x[0], x[0], x[0], x[0]), 1)
        x_ceus_16 = torch.cat((x[1], x[1], x[1], x[1], x[1], x[1], x[1], x[1]), 1)
        x_us_16=torch.add(out[0], x_us_16)
        x_ceus_16=torch.add(out[1], x_ceus_16)
        # x_us_16=torch.add(out[0], x0[0])
        # x_ceus_16=torch.add(out[1], x0[1])
        x16=[x_us_16,x_ceus_16]
        out = self.relu1(x16)
        return out


class DownTransition(nn.Module):
    def __init__(self, inChans, nConvs, elu, dropout=False):
        super(DownTransition, self).__init__()
        outChans = 2*inChans
        self.down_conv = ModuleParallel(nn.Conv3d(inChans, outChans, kernel_size=(1,2,2), stride=(1,2,2)))
        self.down_conv1 = ModuleParallel(nn.Conv3d(outChans, outChans, kernel_size=(3,1,1), stride=(1,1,1),padding=(1,0,0)))
        # self.bn1 = ContBatchNorm3d(outChans)
        self.bn1 = BatchNorm3dParallel(outChans,num_parallel=2)
        self.do1 = passthrough
        self.relu1 = ModuleParallel(ELUCons(elu, outChans))
        self.relu2 = ModuleParallel(ELUCons(elu, outChans))
        if dropout:
            self.do1 = ModuleParallel(nn.Dropout3d())
        self.ops = _make_nConv(outChans, nConvs, elu)

        self.exchange = Exchange()
        self.bn_threshold = 2e-2
        self.bn1_list = []
        for module in self.bn1.modules():
            if isinstance(module, nn.BatchNorm3d):
                self.bn1_list.append(module)

    def forward(self, x):
        # down = self.relu1(self.bn1(self.down_conv1(self.down_conv(x))))
        out = self.bn1(self.down_conv1(self.down_conv(x)))
        out=self.relu1(out)
        if len(x) > 1:
            down = self.exchange(out, self.bn1_list, self.bn_threshold)
        out = self.do1(down)
        out = self.ops(out)
        out0=torch.add(out[0], down[0])
        out00=torch.add(out[1], down[1])
        out=[out0,out00]
        # out = self.relu2(torch.add(out, down))
        out = self.relu2(out)
        return out


class UpTransition(nn.Module):
    def __init__(self, inChans, outChans, nConvs, elu, dropout=False):
        super(UpTransition, self).__init__()
        self.up_conv = ModuleParallel(nn.ConvTranspose3d(inChans, outChans // 2, kernel_size=(1,2,2), stride=(1,2,2)))
        self.up_conv1 = ModuleParallel(nn.ConvTranspose3d(outChans // 2, outChans // 2, kernel_size=(3,1,1), stride=(1,1,1),padding=(1,0,0)))
        # self.bn1 = ContBatchNorm3d(outChans // 2)
        self.bn1 = BatchNorm3dParallel(outChans // 2,num_parallel=2)
        self.do1 = passthrough
        self.do2 = ModuleParallel(nn.Dropout3d())
        self.relu1 =ModuleParallel( ELUCons(elu, outChans // 2))
        self.relu2 = ModuleParallel(ELUCons(elu, outChans))
        if dropout:
            self.do1 =ModuleParallel( nn.Dropout3d())
        self.ops = _make_nConv(outChans, nConvs, elu)

    def forward(self, x, skipx):
        out = self.do1(x)
        skipxdo = self.do2(skipx)
        out = self.relu1(self.bn1(self.up_conv1(self.up_conv(out))))
        # xcat = torch.cat((out, skipxdo), 1)
        xcat0 = torch.cat((out[0], skipxdo[0]), 1)
        xcat1 = torch.cat((out[1], skipxdo[1]), 1)
        xcat=[xcat0,xcat1]
        out = self.ops(xcat)
        out0=torch.add(out[0], xcat[1])
        out00=torch.add(out[1], xcat[1])
        out=[out0,out00]
        # out = self.relu2(torch.add(out, xcat))
        out = self.relu2(out)
        return out


class OutputTransition(nn.Module):
    def __init__(self, inChans, elu, nll):
        super(OutputTransition, self).__init__()
        self.conv1 = ModuleParallel(nn.Conv3d(inChans, 2, kernel_size=5, padding=2))
        # self.bn1 = ContBatchNorm3d(2)
        self.bn1 =BatchNorm3dParallel(2,2)
        self.conv2 = ModuleParallel(nn.Conv3d(2, 2, kernel_size=1))
        self.relu1 =ModuleParallel( ELUCons(elu, 2))
        # self.seg=ModuleParallel(nn.Conv3d(2,2,kernel_size=(4,1,1),stride=1,padding=0))
        

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out0=torch.mean(out[0], 2) + torch.max(out[0], 2)[0]
        out1=torch.mean(out[1], 2) + torch.max(out[1], 2)[0]
        # out=self.seg(out)
        # out0=out[0].squeeze(dim=2)
        # out1=out[1].squeeze(dim=2)
        out=[out0,out1]
        # make channels the last axis\
        return out
        # return out.permute(1,0,2,3)

def swish(x, inplace: bool = False):
    """Swish - Described in: https://arxiv.org/abs/1710.05941
    """
    return x.mul_(x.sigmoid()) if inplace else x.mul(x.sigmoid())
class Swish(nn.Module):
    def __init__(self, inplace: bool = False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return swish(x, self.inplace)
class InnerBlock(nn.Module):
    def __init__(self, dim, kernel_size, project_dim=2):
        super(InnerBlock, self).__init__()

        self.project_dim = project_dim
        self.dim = dim
        self.kernel_size = kernel_size

        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, self.kernel_size, stride=1, padding=self.kernel_size//2, groups=4, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

        share_planes = 8
        factor = 2
        self.embed = nn.Sequential(
            nn.Conv2d(2*dim, dim//factor, 1, bias=False),
            nn.BatchNorm2d(dim//factor),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//factor, dim, kernel_size=1),
            nn.GroupNorm(num_groups=dim // share_planes, num_channels=dim)
        )

        self.conv1x1 = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            nn.BatchNorm3d(dim)
        )

        self.bn = nn.BatchNorm3d(dim)
        self.act = Swish(inplace=True)

        reduction_factor = 4
        self.radix = 2
        attn_chs = max(dim * self.radix // reduction_factor, 32)
        self.se = nn.Sequential(
            nn.Conv3d(dim, attn_chs, 1),
            nn.BatchNorm3d(attn_chs),
            nn.ReLU(inplace=True),
            nn.Conv3d(attn_chs, self.radix*dim, 1)
        )
        self.ttt=nn.Conv3d(dim,dim,(4,1,1),1)
    def forward(self, x):
        # kl=self.ttt(x).squeeze(dim=self.project_dim)
        kl=torch.mean(x, self.project_dim)
        k = kl + torch.max(x, self.project_dim)[0]
        k = self.key_embed(k)
        q = kl + torch.max(x, self.project_dim)[0]
        qk = torch.cat([q, k], dim=1)

        w = self.embed(qk)
        w = w.unsqueeze(self.project_dim)
        fill_shape = w.shape[-1]
        repeat_shape = [1,1,1,1,1]
        repeat_shape[self.project_dim] = fill_shape
        w = w.repeat(repeat_shape)
        
        v = self.conv1x1(x)
        v = v * w
        v = self.bn(v)
        v = self.act(v)

        B, C,T, H, W = v.shape
        v = v.view(B, C, 1,T, H, W)
        x = x.view(B, C, 1, T,H, W)
        x = torch.cat([x, v], dim=2)

        x_ga = x.sum(dim=2)
        x_gap = x_ga.mean((2, 3, 4), keepdim=True)

        x_attn = self.se(x_gap)
        x_attn = x_attn.view(B, C, self.radix)
        x_attn = F.softmax(x_attn, dim=2)
        out = (x * x_attn.reshape((B, C, self.radix, 1, 1, 1))).sum(dim=2)
        x_t = x_ga.mean((3, 4), keepdim=True)
        x_t = x_t.view(B, C, T)
        x_tw = F.softmax(x_t, dim=2)
        out=out*x_tw.reshape((B, C, T, 1, 1))
        return out.contiguous(),x_tw


class VNet(nn.Module):
    # the number of convolutions in each layer corresponds
    # to what is in the actual prototxt, not the intent
    def __init__(self, elu=True, nll=False):
        super(VNet, self).__init__()
        self.in_tr = InputTransition(8, elu)
        self.down_tr32 = DownTransition(8, 1, elu)   #8
        self.down_tr64 = DownTransition(16, 2, elu)
        self.down_tr128 = DownTransition(32, 3, elu, dropout=True)
        self.down_tr256 = DownTransition(64, 2, elu, dropout=True)
        self.up_tr256 = UpTransition(128,128, 2, elu, dropout=True)
        self.up_tr128 = UpTransition(128, 64, 2, elu, dropout=True)
        self.up_tr64 = UpTransition(64, 32, 1, elu)
        self.up_tr32 = UpTransition(32, 16, 1, elu)
        self.out_tr = OutputTransition(16, elu, nll)
        # self.conv = ModuleParallel(nn.Conv3d(3, 1, kernel_size=7, stride=2, padding=3,  bias=False))
        self.conv = ModuleParallel(nn.Conv3d(3, 1, kernel_size=1, stride=1, padding=0,  bias=False))
        # self.conv=nn.Conv3d(3,1,1)
        self.tem = nn.Sequential(
            nn.Conv3d(128, 128, 1, padding=0),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True))
        self.temporal=InnerBlock(dim=128, kernel_size=3, project_dim=2)
    def forward(self, x):
        x=self.conv(x)
        out16 = self.in_tr(x)
        out32 = self.down_tr32(out16)
        out64 = self.down_tr64(out32)
        out128 = self.down_tr128(out64)
        out256 = self.down_tr256(out128)   #16,256
        out256_1=self.tem(out256[1])
        out256_1,x_tw=self.temporal(out256_1)
        out256=[out256[0],out256_1]
        out = self.up_tr256(out256, out128)
        out = self.up_tr128(out, out64)
        out = self.up_tr64(out, out32)
        out = self.up_tr32(out, out16)
        out = self.out_tr(out)
        return out,x_tw,out64
