from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
import numpy as np
import cv2 as cv
from resnet import BasicBlock as ResBlock
from resnet import GatedSpatialConv2d as GatedSpatialConv2d
from scipy.ndimage.morphology import distance_transform_edt
from wider_resnet import wider_resnet38_a2
from mynn import initialize_weights, Norm2d, Upsample
import logging


class conv_block(nn.Module):
    """
    Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class down_conv(nn.Module):
    """
    Down Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(down_conv, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.down(x)
        return x

class sr_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(sr_conv, self).__init__()
        self.sr = nn.Sequential(
            nn.Conv2d(in_ch, 4 * in_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.PixelShuffle(2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.sr(x)
        return x


def mask_to_onehot(mask, num_classes=8):
    _mask = [mask == i for i in range(1, num_classes+1)]
    _mask = [np.expand_dims(x, 0) for x in _mask]
    return np.concatenate(_mask, 0)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class SFTLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(SFTLayer, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.SFT_scale_conv0 = nn.Conv2d(self.in_ch, self.out_ch, 1)
        self.SFT_scale_conv1 = nn.Conv2d(self.out_ch, self.out_ch, 1)
        self.SFT_shift_conv0 = nn.Conv2d(self.in_ch, self.out_ch, 1)
        self.SFT_shift_conv1 = nn.Conv2d(self.out_ch, self.out_ch, 1)

    def forward(self, x, y):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(y), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(y), 0.1, inplace=True))
        return x * (scale + 1) + shift



class edge_Net(nn.Module):

    def __init__(self, args):
        super(edge_Net, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])
        ################################################################################

        self.Eg5 = up_conv(filters[4], filters[3])
        self.Eg_conv5 = conv_block(filters[3], filters[3])

        self.Eg4 = up_conv(filters[3], filters[2])
        self.Eg_conv4 = conv_block(filters[2], filters[2])

        self.Eg3 = up_conv(filters[2], filters[1])
        self.Eg_conv3 = conv_block(filters[1], filters[1])

        self.Eg2 = up_conv(filters[1], filters[0])
        self.Eg_conv2 = conv_block(filters[0], filters[0])

        ###############################################################################

        self.Conv_Edge = nn.Conv2d(filters[0], 1, kernel_size=1, stride=1, padding=0)

        self.gate4 = GatedSpatialConv2d(512, 512)
        self.gate3 = GatedSpatialConv2d(256, 256)
        self.gate2 = GatedSpatialConv2d(128, 128)
        self.gate1 = GatedSpatialConv2d(64, 64)
        self.sa4 = self.sa3 = self.sa2 =self.sa1 = SpatialAttention()


        self.co = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

        ################################################################################
        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])


        self.Conv_down4 = nn.Conv2d(filters[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down3 = nn.Conv2d(filters[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down2 = nn.Conv2d(filters[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down1 = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

        self.Conv_d4 = self.Conv_d3 = self.Conv_d2 =self.Conv_d1 =nn.Conv2d(num_classes, 1, kernel_size=1, stride=1, padding=0)
        self.Conv_Seg = nn.Conv2d(56, num_classes, kernel_size=1, stride=1, padding=0)
        ################################################################################

    def forward(self, x):
        x_size = x.size()
       
        #encoder
        e1 = self.Conv1(x) #64
        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #128
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) # 256
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) # 512
        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # 1024

        # seg decoder
        d5 = self.Up5(e5) #512
        d5 = torch.cat((e4, d5), dim=1) #1024
        d5 = self.Up_conv5(d5) #512
        d5_1 = self.Conv_down4(d5) #8
        d5_1 = self.sa4(d5_1) * d5_1
        gate4 = self.Conv_d4(d5_1) #1

        d4 = self.Up4(d5) #256
        d4 = torch.cat((e3, d4), dim=1) #512
        d4 = self.Up_conv4(d4) #256
        d4_1 = self.Conv_down3(d4)  # 8
        d4_1 = self.sa3(d4_1) * d4_1
        gate3 = self.Conv_d3(d4_1) #1

        d3 = self.Up3(d4) #128     
        d3 = torch.cat((e2, d3), dim=1) #256
        d3 = self.Up_conv3(d3) #128
        d3_1 = self.Conv_down2(d3)  # 8
        d3_1 = self.sa2(d3_1) * d3_1
        gate2 = self.Conv_d2(d3_1) #1

        d2 = self.Up2(d3) #64
        d2 = torch.cat((e1, d2), dim=1) #128
        d2 = self.Up_conv2(d2) #64
        d2_1 = self.Conv_down1(d2)  # 8    
        d2_1 = self.sa1(d2_1) * d2_1
        gate1 = self.Conv_d1(d2_1) #1
        
        
        #edge decoder
        g5 = self.Eg5(e5) #512
        g5 = self.Eg_conv5(g5) #512
        g5 = self.gate4(g5, gate4)

        g4 = self.Eg4(g5) #256
        g4 = self.Eg_conv4(g4) #256
        g4 = self.gate3(g4, gate3)

        g3 = self.Eg3(g4) #128
        g3 = self.Eg_conv3(g3) #128
        g3 = self.gate2(g3, gate2)

        g2 = self.Eg2(g3) #64
        g2 = self.Eg_conv2(g2) #64
        g2 = self.gate1(g2, gate1) 
         
        g1 = self.Conv_Edge(g2) #1
        g0 = F.interpolate(g1, x_size[2:], mode='bilinear', align_corners=True)  # 上采样
        out_Edge = self.sigmoid(g0)  # 8 1 176 160 
########################################################
        o4 = F.interpolate(d5_1, x_size[2:], mode='bilinear', align_corners=True)
        o3 = F.interpolate(d4_1, x_size[2:], mode='bilinear', align_corners=True)
        o2 = F.interpolate(d3_1, x_size[2:], mode='bilinear', align_corners=True)
        o1 = F.interpolate(d2_1, x_size[2:], mode='bilinear', align_corners=True)


        edge_1 = edge_2 = edge_3 = edge_4 = out_Edge 
        o0 = torch.cat((o4, edge_4, o3, edge_3, o2, edge_2, o1, edge_1), dim=1)
        
        out_Seg = self.Conv_Seg(o0) #7
        return out_Seg, out_Edge#, o4,o3,o2,o1#



class supv_UNet(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(supv_UNet, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        
        self.Conv_down4 = nn.Conv2d(filters[3], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down3 = nn.Conv2d(filters[2], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down2 = nn.Conv2d(filters[1], num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down1 = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Conv_Seg = nn.Conv2d(28, num_classes, kernel_size=1, stride=1, padding=0)


    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x_size = x.size()
       
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)
        d5_1 = self.Conv_down4(d5) #8

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        d4_1 = self.Conv_down3(d4)  # 8

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)
        d3_1 = self.Conv_down2(d3)  # 8

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2_1 = self.Conv_down1(d2)  # 8   
        
        o4 = F.interpolate(d5_1 , x_size[2:], mode='bilinear', align_corners=True)
        o3 = F.interpolate(d4_1 , x_size[2:], mode='bilinear', align_corners=True)
        o2 = F.interpolate(d3_1 , x_size[2:], mode='bilinear', align_corners=True)
        o1 = F.interpolate(d2_1 , x_size[2:], mode='bilinear', align_corners=True)
        
        o0 = torch.cat((o4, o3, o2, o1), dim=1)
        
        out = self.Conv_Seg(o0) #7
        return out, o4,o3,o2,o1#       


class shape_Unet(nn.Module):
    def __init__(self, args):
        super(shape_Unet, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        
        self.Up_conv1 = conv_block(filters[1], filters[0])
        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    # Shape Stream
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

        self.c3 = nn.Conv2d(128, 1, kernel_size=1)
        self.c4 = nn.Conv2d(256, 1, kernel_size=1)
        self.c5 = nn.Conv2d(512, 1, kernel_size=1)

        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)
        self.d3 = nn.Conv2d(8, 4, kernel_size=1)
        self.fuse = nn.Conv2d(4, 1, kernel_size=1, padding=0, bias=False)

        self.cw = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)

        self.gate1 = GatedSpatialConv2d(16, 16)
        self.gate2 = GatedSpatialConv2d(8, 8)
        self.gate3 = GatedSpatialConv2d(4, 4)

        self.expand = nn.Sequential(nn.Conv2d(1, 64, kernel_size=1),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x_size = x.size()
        # Encoder
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)
        # Shape Stream
        ss = F.interpolate(self.d0(e1), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.res1(ss)
        c3 = F.interpolate(self.c3(e2), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.d1(ss)
        ss = self.gate1(ss, c3)
        ss = self.res2(ss)
        ss = self.d2(ss)
        c4 = F.interpolate(self.c4(e3), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.gate2(ss, c4)
        ss = self.res3(ss)
        ss = self.d3(ss)
        c5 = F.interpolate(self.c5(e4), x_size[2:],
                           mode='bilinear', align_corners=True)
        ss = self.gate3(ss, c5)
        ss = self.fuse(ss)
        ss = F.interpolate(ss, x_size[2:], mode='bilinear', align_corners=True)
        edge_out = self.sigmoid(ss)

        ### Canny Edge
        im_arr = np.mean(x.cpu().numpy(), axis=1).astype(np.uint8)
        canny = np.zeros((x_size[0], 1, x_size[2], x_size[3]))
        for i in range(x_size[0]):
            canny[i] = cv.Canny(im_arr[i], 10, 100)
        canny = torch.from_numpy(canny).cuda().float()
        ### End Canny Edge

        #cat = torch.cat([edge_out, canny], dim=1)
        #acts = self.cw(cat)
        #acts = self.sigmoid(acts)
        edge = self.expand(edge_out)

        # Decoder
        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = torch.cat([d2, edge], dim=1)
        d1 = self.Up_conv1(d1)
        out = self.Conv(d1)

        # d1 = self.active(out)

        return out, edge_out


class U_Net(nn.Module):
    """
    UNet - Basic Implementation
    Paper : https://arxiv.org/abs/1505.04597
    """

    def __init__(self, args):
        super(U_Net, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])
        
        
        #self.Conv_down4 = nn.Conv2d(filters[3], num_classes, kernel_size=1, stride=1, padding=0)
        #self.Conv_down3 = nn.Conv2d(filters[2], num_classes, kernel_size=1, stride=1, padding=0)
        #self.Conv_down2 = nn.Conv2d(filters[1], num_classes, kernel_size=1, stride=1, padding=0)
        #self.Conv_down1 = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)
        
        #self.Conv_seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        x_size = x.size()
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_conv5(d5)
       
        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)
        
        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        
        #seg_5 = self.Conv_down4(F.interpolate(d5, x_size[2:], mode='bilinear', align_corners=True))
        #seg_4 = self.Conv_down3(F.interpolate(d4, x_size[2:], mode='bilinear', align_corners=True))
        #seg_3 = self.Conv_down2(F.interpolate(d3, x_size[2:], mode='bilinear', align_corners=True))
        #seg_2 = self.Conv_down(d2)
        #seg_out = self.Conv_seg(torch.cat((seg_5, seg_4, seg_3, seg_2), dim=1))


        out = self.Conv(d2)

        # d1 = self.active(out)

        return out  #seg_out,seg_2 


class decouple_Net(nn.Module):

    def __init__(self, args):
        super(decouple_Net, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv0 = nn.Conv2d(in_ch, 32, kernel_size=1)
        
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        
        self.de1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.dn = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)

        self.Conv_down5 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Edge_down5 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)


        self.Conv_edge = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.Conv_Seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        
               
        self.squeeze_body_edge5 = SqueezeBodyEdge(512, Norm2d)
        self.squeeze_body_edge4 = SqueezeBodyEdge(256, Norm2d)
        self.squeeze_body_edge3 = SqueezeBodyEdge(128, Norm2d)
        self.squeeze_body_edge = SqueezeBodyEdge(64, Norm2d)

        self.sigmoid_edge = nn.Sigmoid()        
 

    def forward(self, x):
        x_size = x.size()
       
        e1 = self.Conv1(x) #64

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #128
                       
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #256
               
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #512

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [8, 1024, 6, 8]
                                                 
        d_5 = self.Up5(e5) #512
        d5 = torch.cat((e4, d_5), dim=1)
        d5 = self.Up_conv5(d5)
        seg_body5, seg_edge5 = self.squeeze_body_edge5(d5,d_5)
        
        seg_add5 = seg_body5+seg_edge5
        d_4 = self.Up4(seg_add5)
        d4 = torch.cat((e3, d_4), dim=1)
        d4 = self.Up_conv4(d4) #256

        seg_body4, seg_edge4 = self.squeeze_body_edge4(d4,d_4)
        seg_add4 = seg_body4+seg_edge4                   
        d_3 = self.Up3(seg_add4)
        d3 = torch.cat((e2, d_3), dim=1)
        d3 = self.Up_conv3(d3)#128

        seg_body3, seg_edge3 = self.squeeze_body_edge3(d3,d_3)
        seg_add3 = seg_body3+seg_edge3
        d_2 = self.Up2(seg_add3)
        d2 = torch.cat((e1, d_2), dim=1)
        d2 = self.Up_conv2(d2) #64  
                 
        seg_body21, seg_edge21 = self.squeeze_body_edge(d2,d_2)#64 
        seg_add2 = seg_body21 + seg_edge21 #64                       
       
        seg_edge5 = self.Edge_down5(F.interpolate(seg_edge5, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge4 = self.Edge_down4(F.interpolate(seg_edge4, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge3 = self.Edge_down3(F.interpolate(seg_edge3, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge_add = self.Edge_down2(torch.cat((seg_edge5, seg_edge4, seg_edge3, seg_edge21), dim=1))#64
        seg_edge = self.d0(seg_edge_add)#32
        seg_edge = self.res1(seg_edge)#32
        seg_edge = self.d1(seg_edge)#16
        seg_edge = self.res2(seg_edge)#16
        seg_edge = self.d2(seg_edge)#8
        seg_edge_refine = self.res3(seg_edge)#8
        seg_edge = self.Conv_edge(seg_edge_refine)#1
        seg_edge_out = self.sigmoid_edge(seg_edge)
        
        seg_body5 = self.Conv_down5(F.interpolate(seg_body5, x_size[2:], mode='bilinear', align_corners=True))
        seg_body4 = self.Conv_down4(F.interpolate(seg_body4, x_size[2:], mode='bilinear', align_corners=True))
        seg_body3 = self.Conv_down3(F.interpolate(seg_body3, x_size[2:], mode='bilinear', align_corners=True))
        seg_body2 = self.Conv_down2(seg_body21)
        seg_body = torch.cat((seg_body5, seg_body4, seg_body3, seg_body2), dim=1)#52
        seg_body_out = self.Conv_Seg(seg_body)
        
        seg_out = seg_body + seg_edge
        seg_final_out = self.final_Seg(seg_out)
         
                                           
        return seg_final_out, seg_body_out, seg_edge_out#, e1, d_2, d2, seg_body21, seg_edge21 

class decouple_multiscale(nn.Module):

    def __init__(self, args):
        super(decouple_multiscale, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv0 = nn.Conv2d(in_ch, 32, kernel_size=1)
        
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        
        self.de1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.dn = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)

        self.Conv_down5 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Edge_down5 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)


        self.Conv_edge = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.Conv_Seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        
               
        self.squeeze_body_edge5 = SqueezeBodyEdge(512, Norm2d)
        self.squeeze_body_edge4 = SqueezeBodyEdge(256, Norm2d)
        self.squeeze_body_edge3 = SqueezeBodyEdge(128, Norm2d)
        self.squeeze_body_edge = SqueezeBodyEdge(64, Norm2d)

        self.sigmoid_edge = nn.Sigmoid()        
 

    def forward(self, x):
        x_size = x.size()
       
        e1 = self.Conv1(x) #64

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #128
                       
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #256
               
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #512

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [8, 1024, 6, 8]
                                                 
        d_5 = self.Up5(e5) #512
        d5 = torch.cat((e4, d_5), dim=1)
        d5 = self.Up_conv5(d5)
        seg_body5, seg_edge5 = self.squeeze_body_edge5(d5,d_5)
        
        seg_add5 = seg_body5+seg_edge5
        d_4 = self.Up4(seg_add5)
        d4 = torch.cat((e3, d_4), dim=1)
        d4 = self.Up_conv4(d4) #256

        seg_body4, seg_edge4 = self.squeeze_body_edge4(d4,d_4)
        seg_add4 = seg_body4+seg_edge4                   
        d_3 = self.Up3(seg_add4)
        d3 = torch.cat((e2, d_3), dim=1)
        d3 = self.Up_conv3(d3)#128

        seg_body3, seg_edge3 = self.squeeze_body_edge3(d3,d_3)
        seg_add3 = seg_body3+seg_edge3
        d_2 = self.Up2(seg_add3)
        d2 = torch.cat((e1, d_2), dim=1)
        d2 = self.Up_conv2(d2) #64  
                 
        seg_body21, seg_edge21 = self.squeeze_body_edge(d2,d_2)#64 
        seg_add2 = seg_body21 + seg_edge21 #64                       
       
        #seg_edge5 = self.Edge_down5(F.interpolate(seg_edge5, x_size[2:], mode='bilinear', align_corners=True))
        #seg_edge4 = self.Edge_down4(F.interpolate(seg_edge4, x_size[2:], mode='bilinear', align_corners=True))
        #seg_edge3 = self.Edge_down3(F.interpolate(seg_edge3, x_size[2:], mode='bilinear', align_corners=True))
        #seg_edge_add = self.Edge_down2(torch.cat((seg_edge5, seg_edge4, seg_edge3, seg_edge21), dim=1))#64
        seg_edge = self.d0(seg_edge21)#32
        seg_edge = self.res1(seg_edge)#32
        seg_edge = self.d1(seg_edge)#16
        seg_edge = self.res2(seg_edge)#16
        seg_edge = self.d2(seg_edge)#8
        seg_edge_refine = self.res3(seg_edge)#8
        seg_edge = self.Conv_edge(seg_edge_refine)#1
        seg_edge_out = self.sigmoid_edge(seg_edge)
        
        seg_body5 = self.Conv_down5(F.interpolate(seg_body5, x_size[2:], mode='bilinear', align_corners=True))
        seg_body4 = self.Conv_down4(F.interpolate(seg_body4, x_size[2:], mode='bilinear', align_corners=True))
        seg_body3 = self.Conv_down3(F.interpolate(seg_body3, x_size[2:], mode='bilinear', align_corners=True))
        seg_body2 = self.Conv_down2(seg_body21)
        seg_body = torch.cat((seg_body5, seg_body4, seg_body3, seg_body2), dim=1)#52
        seg_body_out = self.Conv_Seg(seg_body)
        
        seg_out = seg_body + seg_edge
        seg_final_out = self.final_Seg(seg_out)
         
                                           
        return seg_final_out, seg_body_out, seg_edge_out#, e1, d_2, d2, seg_body21, seg_edge21 

class decouple_body(nn.Module):

    def __init__(self, args):
        super(decouple_body, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv0 = nn.Conv2d(in_ch, 32, kernel_size=1)
        
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        
        self.de1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.dn = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)

        self.Conv_down5 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Edge_down5 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down2 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)


        self.Conv_edge = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.Conv_Seg = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
               
        self.squeeze_body_edge5 = SqueezeBodyEdge(512, Norm2d)
        self.squeeze_body_edge4 = SqueezeBodyEdge(256, Norm2d)
        self.squeeze_body_edge3 = SqueezeBodyEdge(128, Norm2d)
        self.squeeze_body_edge = SqueezeBodyEdge(64, Norm2d)

        self.sigmoid_edge = nn.Sigmoid()        
 

    def forward(self, x):
        x_size = x.size()
       
        e1 = self.Conv1(x) #64

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #128
                       
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #256
               
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #512

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [8, 1024, 6, 8]
                                                 
        d_5 = self.Up5(e5) #512
        d5 = torch.cat((e4, d_5), dim=1)
        d5 = self.Up_conv5(d5)
        seg_body5, seg_edge5 = self.squeeze_body_edge5(d5,d_5)
        
        seg_add5 = seg_body5+seg_edge5
        d_4 = self.Up4(seg_add5)
        d4 = torch.cat((e3, d_4), dim=1)
        d4 = self.Up_conv4(d4) #256

        seg_body4, seg_edge4 = self.squeeze_body_edge4(d4,d_4)
        seg_add4 = seg_body4+seg_edge4                   
        d_3 = self.Up3(seg_add4)
        d3 = torch.cat((e2, d_3), dim=1)
        d3 = self.Up_conv3(d3)#128

        seg_body3, seg_edge3 = self.squeeze_body_edge3(d3,d_3)
        seg_add3 = seg_body3+seg_edge3
        d_2 = self.Up2(seg_add3)
        d2 = torch.cat((e1, d_2), dim=1)
        d2 = self.Up_conv2(d2) #64  
                 
        seg_body21, seg_edge21 = self.squeeze_body_edge(d2,d_2)#64 
        seg_add2 = seg_body21 + seg_edge21 #64                       
       
        seg_edge5 = self.Edge_down5(F.interpolate(seg_edge5, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge4 = self.Edge_down4(F.interpolate(seg_edge4, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge3 = self.Edge_down3(F.interpolate(seg_edge3, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge_add = self.Edge_down2(torch.cat((seg_edge5, seg_edge4, seg_edge3, seg_edge21), dim=1))#64
        seg_edge = self.d0(seg_edge_add)#32
        seg_edge = self.res1(seg_edge)#32
        seg_edge = self.d1(seg_edge)#16
        seg_edge = self.res2(seg_edge)#16
        seg_edge = self.d2(seg_edge)#8
        seg_edge_refine = self.res3(seg_edge)#8
        seg_edge = self.Conv_edge(seg_edge_refine)#1
        seg_edge_out = self.sigmoid_edge(seg_edge)
        
        #seg_body5 = self.Conv_down5(F.interpolate(seg_body5, x_size[2:], mode='bilinear', align_corners=True))
        #seg_body4 = self.Conv_down4(F.interpolate(seg_body4, x_size[2:], mode='bilinear', align_corners=True))
        #seg_body3 = self.Conv_down3(F.interpolate(seg_body3, x_size[2:], mode='bilinear', align_corners=True))
        #seg_body2 = self.Conv_down2(seg_body21)
        #seg_body = torch.cat((seg_body5, seg_body4, seg_body3, seg_body2), dim=1)#52
        seg_body_out = self.Conv_Seg(seg_body21)
        
        seg_out = seg_body21 + seg_edge
        seg_final_out = self.final_Seg(seg_out)
         
                                           
        return seg_final_out, seg_body_out, seg_edge_out#, e1, d_2, d2, seg_body21, seg_edge21 

class decouple_res(nn.Module):

    def __init__(self, args):
        super(decouple_res, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.Conv0 = nn.Conv2d(in_ch, 32, kernel_size=1)
        
        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        
        self.de1 = nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0)
        self.dn = nn.Conv2d(96, 64, kernel_size=1, stride=1, padding=0)
        self.d0 = nn.Conv2d(64, 32, kernel_size=1)
        self.res1 = ResBlock(32, 32)
        self.d1 = nn.Conv2d(32, 16, kernel_size=1)
        self.res2 = ResBlock(16, 16)
        self.d2 = nn.Conv2d(16, 8, kernel_size=1)
        self.res3 = ResBlock(8, 8)

        self.Conv_down5 = nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down4 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down3 = nn.Conv2d(128, num_classes, kernel_size=1, stride=1, padding=0)
        self.Conv_down2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        
        self.Edge_down5 = nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down4 = nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down3 = nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0)
        self.Edge_down2 = nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0)


        self.Conv_edge = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        self.Conv_Seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg2 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        self.final_Seg = nn.Conv2d(52, num_classes, kernel_size=1, stride=1, padding=0)
        
               
        self.squeeze_body_edge5 = SqueezeBodyEdge(512, Norm2d)
        self.squeeze_body_edge4 = SqueezeBodyEdge(256, Norm2d)
        self.squeeze_body_edge3 = SqueezeBodyEdge(128, Norm2d)
        self.squeeze_body_edge = SqueezeBodyEdge(64, Norm2d)

        self.sigmoid_edge = nn.Sigmoid()        
 

    def forward(self, x):
        x_size = x.size()
       
        e1 = self.Conv1(x) #64

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2) #128
                       
        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3) #256
               
        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4) #512

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5) # [8, 1024, 6, 8]
                                                 
        d_5 = self.Up5(e5) #512
        d5 = torch.cat((e4, d_5), dim=1)
        d5 = self.Up_conv5(d5)
        seg_body5, seg_edge5 = self.squeeze_body_edge5(d5,d_5)
        
        seg_add5 = seg_body5+seg_edge5
        d_4 = self.Up4(seg_add5)
        d4 = torch.cat((e3, d_4), dim=1)
        d4 = self.Up_conv4(d4) #256

        seg_body4, seg_edge4 = self.squeeze_body_edge4(d4,d_4)
        seg_add4 = seg_body4+seg_edge4                   
        d_3 = self.Up3(seg_add4)
        d3 = torch.cat((e2, d_3), dim=1)
        d3 = self.Up_conv3(d3)#128

        seg_body3, seg_edge3 = self.squeeze_body_edge3(d3,d_3)
        seg_add3 = seg_body3+seg_edge3
        d_2 = self.Up2(seg_add3)
        d2 = torch.cat((e1, d_2), dim=1)
        d2 = self.Up_conv2(d2) #64  
                 
        seg_body21, seg_edge21 = self.squeeze_body_edge(d2,d_2)#64 
        seg_add2 = seg_body21 + seg_edge21 #64                       
       
        seg_edge5 = self.Edge_down5(F.interpolate(seg_edge5, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge4 = self.Edge_down4(F.interpolate(seg_edge4, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge3 = self.Edge_down3(F.interpolate(seg_edge3, x_size[2:], mode='bilinear', align_corners=True))
        seg_edge = self.Edge_down2(torch.cat((seg_edge5, seg_edge4, seg_edge3, seg_edge21), dim=1))#1
        #seg_edge = self.d0(seg_edge_add)#32
        #seg_edge = self.res1(seg_edge)#32
        #seg_edge = self.d1(seg_edge)#16
        #seg_edge = self.res2(seg_edge)#16
        #seg_edge = self.d2(seg_edge)#8
        #seg_edge_refine = self.res3(seg_edge)#8
        #seg_edge = self.Conv_edge(seg_edge_refine)#1
        seg_edge_out = self.sigmoid_edge(seg_edge)
        
        seg_body5 = self.Conv_down5(F.interpolate(seg_body5, x_size[2:], mode='bilinear', align_corners=True))
        seg_body4 = self.Conv_down4(F.interpolate(seg_body4, x_size[2:], mode='bilinear', align_corners=True))
        seg_body3 = self.Conv_down3(F.interpolate(seg_body3, x_size[2:], mode='bilinear', align_corners=True))
        seg_body2 = self.Conv_down2(seg_body21)
        seg_body = torch.cat((seg_body5, seg_body4, seg_body3, seg_body2), dim=1)#52
        seg_body_out = self.Conv_Seg(seg_body)
        
        seg_out = seg_body + seg_edge
        seg_final_out = self.final_Seg(seg_out)
         
                                           
        return seg_final_out, seg_body_out, seg_edge_out#, e1, d_2, d2, seg_body21, seg_edge21 
class Decouple(nn.Module):
    """
    WideResNet38 version of DeepLabV3
    mod1
    pool2
    mod2 bot_fine
    pool3
    mod3-7
    bot_aspp

    structure: [3, 3, 6, 3, 1, 1],[3, 3, 6, 3, 1, 1]
    channels = [(64, 64),(128, 128), (256, 256), (256, 512),
                    (256,512, 1024), (512, 1024, 2048)]
    """

    def __init__(self, args, trunk='WideResnet38', criterion=None):

        super(Decouple, self).__init__()
        num_classes = args['num_classes']
        self.criterion = criterion
        logging.info("Trunk: %s", trunk)

        wide_resnet = wider_resnet38_a2(classes=1000, dilation=True)
        wide_resnet = torch.nn.DataParallel(wide_resnet)
        if criterion is not None:
            try:
                checkpoint = torch.load('./pretrained_models/wider_resnet20.pth.tar', map_location='cpu')
                wide_resnet.load_state_dict(checkpoint)
                del checkpoint
            except:
                print("Please download the ImageNet weights of WideResNet20 in our repo to ./pretrained_models/wider_resnet38.pth.tar.")
                raise RuntimeError("=====================Could not load ImageNet weights of WideResNet38 network.=======================")

        wide_resnet = wide_resnet.module

        self.mod1 = wide_resnet.mod1
        self.mod2 = wide_resnet.mod2
        self.mod3 = wide_resnet.mod3
        self.mod4 = wide_resnet.mod4
        self.mod5 = wide_resnet.mod5
        self.mod6 = wide_resnet.mod6
        self.mod7 = wide_resnet.mod7
        self.pool2 = wide_resnet.pool2
        self.pool3 = wide_resnet.pool3
        del wide_resnet

        self.aspp = ASPP(4096, 256, output_stride=8)

        self.bot_aspp = nn.Conv2d(1280, 256, kernel_size=1, bias=False)

        self.bot_fine = nn.Conv2d(128, 48, kernel_size=1, bias=False)

        edge_dim = 256
        self.edge_conv = nn.Sequential(
            nn.Conv2d(1, edge_dim, kernel_size=1, bias=False),
            Norm2d(edge_dim), nn.ReLU(inplace=True))

        self.squeeze_body_edge = SqueezeBodyEdge(256, Norm2d)
        # fusion different edges
        self.edge_fusion = nn.Conv2d(256 + 48, 256,1,bias=False)
        self.sigmoid_edge = nn.Sigmoid()

        self.edge_out = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=3, padding=1, bias=False),
            Norm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 1, kernel_size=1, bias=False)
        )

        self.dsn_seg_body = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

        self.final_seg = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            Norm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False))

        initialize_weights(self.final_seg, self.dsn_seg_body)

    def forward(self, inp, gts=None):

        x_size = inp.size()
        x = self.mod1(inp)
        m2 = self.mod2(self.pool2(x))
        fine_size = m2.size()
        x = self.mod3(self.pool3(m2))
        x = self.mod4(x)
        x = self.mod5(x)
        x = self.mod6(x)
        x = self.mod7(x)
        x = self.aspp(x)
        aspp = self.bot_aspp(x)


        seg_body, seg_edge = self.squeeze_body_edge(aspp)

        # may add canny edge
        # canny_edge = self.edge_canny(inp, x_size)
        # add low-level feature
        dec0_fine = self.bot_fine(m2)
        seg_edge = self.edge_fusion(torch.cat([Upsample(seg_edge, fine_size[2:]), dec0_fine], dim=1))
        seg_edge_out = self.edge_out(seg_edge)


        seg_out = seg_edge + Upsample(seg_body, fine_size[2:])
        aspp = Upsample(aspp, fine_size[2:])

        seg_out = torch.cat([aspp, seg_out],dim=1)
        seg_final = self.final_seg(seg_out)

        seg_edge_out = Upsample(seg_edge_out, x_size[2:])
        seg_edge_out = self.sigmoid_edge(seg_edge_out)

        seg_final_out = Upsample(seg_final, x_size[2:])

        seg_body_out = Upsample(self.dsn_seg_body(seg_body), x_size[2:])

        # dec0_up = Upsample(dec0_up, m2.size()[2:])
        # dec0 = [dec0_fine, dec0_up]
        # dec0 = torch.cat(dec0, 1)
        # dec1 = self.final(dec0)
        # seg_out = Upsample(dec1, x_size[2:])

        #if self.training:
        return seg_final_out, seg_body_out, seg_edge_out

        #return seg_final_out
        

class ASPP(nn.Module):
    """
    operations performed:
      1x1 x depth
      3x3 x depth dilation 6
      3x3 x depth dilation 12
      3x3 x depth dilation 18
      image pooling
      concatenate all together
      Final 1x1 conv
    """

    def __init__(self, in_dim, reduction_dim=256, output_stride=16, rates=(6, 12, 18)):
        super(ASPP, self).__init__()

        if output_stride == 8:
            rates = [2 * r for r in rates]
        elif output_stride == 16:
            pass
        else:
            raise 'output stride of {} not supported'.format(output_stride)

        self.features = []
        # 1x1
        self.features.append(
            nn.Sequential(nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                          Norm2d(reduction_dim), nn.ReLU(inplace=True)))
        # other rates
        for r in rates:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3,
                          dilation=r, padding=r, bias=False),
                Norm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = torch.nn.ModuleList(self.features)

        # img level features
        self.img_pooling = nn.AdaptiveAvgPool2d(1)
        self.img_conv = nn.Sequential(
            nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
            Norm2d(reduction_dim), nn.ReLU(inplace=True))

    def forward(self, x):
        x_size = x.size()

        img_features = self.img_pooling(x)
        img_features = self.img_conv(img_features)
        img_features = Upsample(img_features, x_size[2:])
        out = img_features

        for f in self.features:
            y = f(x)
            out = torch.cat((out, y), 1)
        return out


class SqueezeBodyEdge(nn.Module):
    def __init__(self, inplane, norm_layer):
        """
        implementation of body generation part
        :param inplane:
        :param norm_layer:
        """
        super(SqueezeBodyEdge, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(inplane, inplane, kernel_size=2, groups=inplane, stride=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplane, inplane, kernel_size=2, groups=inplane, stride=1),
            norm_layer(inplane),
            nn.ReLU(inplace=True)
        )

        self.flow_make = nn.Conv2d(inplane *2 , 2, kernel_size=1, padding=0, bias=False)

    def forward(self, x,seg_down):
        size = x.size()[2:]        
        #seg_down = self.down(x)
        seg_down = F.interpolate(seg_down, size=size, mode="bilinear", align_corners=True)
        flow = self.flow_make(torch.cat([x, seg_down], dim=1))  #shape【B,2,H.W】，2为x y两个方向的偏移       
        seg_flow_warp = self.flow_warp(x, flow, size)
        seg_edge = x - seg_flow_warp
        return seg_flow_warp, seg_edge

    def flow_warp(self, input, flow, size):
        out_h, out_w = size
        n, c, h, w = input.size()

        norm = torch.tensor([[[[out_w, out_h]]]]).type_as(input).to(input.device)
        # new
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w) #-1~1按h等分，排成一维数据，复制成一列，w行  
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2) #增加新维度， 在新维度上concat
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)#新增一个维度 复制n份grid 转换成input数据类型 ，复制到input所在cpu 
        #首先构造一个恒等采样矩阵grid（左上角 是（-1，1）右下角是（1，1） 
        grid = grid + flow.permute(0, 2, 3, 1) / norm #flow.permute(0, 2, 3, 1) / norm 是以像素为单位的offset数组 
        #然后在 grid 基础上增加x，y 方向的 offset，构成一个新的采样矩阵 grid
        output = F.grid_sample(input, grid) #利用grid采样矩阵对原图进行采样 得到变换后的新特征 
        return output


class Recurrent_block(nn.Module):
    """
    Recurrent Block for R2Unet_CNN
    """

    def __init__(self, out_ch, t=2):
        super(Recurrent_block, self).__init__()

        self.t = t
        self.out_ch = out_ch
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x = self.conv(x)
            out = self.conv(x + x)
        return out


class RRCNN_block(nn.Module):
    """
    Recurrent Residual Convolutional Neural Network Block
    """

    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNN_block, self).__init__()

        self.RCNN = nn.Sequential(
            Recurrent_block(out_ch, t=t),
            Recurrent_block(out_ch, t=t)
        )
        self.Conv = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.Conv(x)
        x2 = self.RCNN(x1)
        out = x1 + x2
        return out


class R2U_Net(nn.Module):
    """
    R2U-Unet implementation
    Paper: https://arxiv.org/abs/1802.06955
    """

    def __init__(self, img_ch=3, output_ch=1, t=2):
        super(R2U_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(img_ch, filters[0], t=t)

        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)

        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)

        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)

        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], output_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool1(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool2(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool3(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        # out = self.active(out)

        return out


class Attention_block(nn.Module):
    """
    Attention Block
    """

    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out


class AttU_Net(nn.Module):
    """
    Attention Unet implementation
    Paper: https://arxiv.org/abs/1804.03999
    """

    def __init__(self, args):
        super(AttU_Net, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], num_classes, kernel_size=1, stride=1, padding=0)

        # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.Conv1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.Conv3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.Conv4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.Conv5(e5)

        # print(x5.shape)
        d5 = self.Up5(e5)
        # print(d5.shape)
        x4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


class R2AttU_Net(nn.Module):
    """
    Residual Recuurent Block with attention Unet
    Implementation : https://github.com/LeeJunHyun/Image_Segmentation
    """

    def __init__(self, in_ch=3, out_ch=1, t=2):
        super(R2AttU_Net, self).__init__()

        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.RRCNN1 = RRCNN_block(in_ch, filters[0], t=t)
        self.RRCNN2 = RRCNN_block(filters[0], filters[1], t=t)
        self.RRCNN3 = RRCNN_block(filters[1], filters[2], t=t)
        self.RRCNN4 = RRCNN_block(filters[2], filters[3], t=t)
        self.RRCNN5 = RRCNN_block(filters[3], filters[4], t=t)

        self.Up5 = up_conv(filters[4], filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_RRCNN5 = RRCNN_block(filters[4], filters[3], t=t)

        self.Up4 = up_conv(filters[3], filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_RRCNN4 = RRCNN_block(filters[3], filters[2], t=t)

        self.Up3 = up_conv(filters[2], filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_RRCNN3 = RRCNN_block(filters[2], filters[1], t=t)

        self.Up2 = up_conv(filters[1], filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=32)
        self.Up_RRCNN2 = RRCNN_block(filters[1], filters[0], t=t)

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

    # self.active = torch.nn.Sigmoid()

    def forward(self, x):
        e1 = self.RRCNN1(x)

        e2 = self.Maxpool1(e1)
        e2 = self.RRCNN2(e2)

        e3 = self.Maxpool2(e2)
        e3 = self.RRCNN3(e3)

        e4 = self.Maxpool3(e3)
        e4 = self.RRCNN4(e4)

        e5 = self.Maxpool4(e4)
        e5 = self.RRCNN5(e5)

        d5 = self.Up5(e5)
        e4 = self.Att5(g=d5, x=e4)
        d5 = torch.cat((e4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        e3 = self.Att4(g=d4, x=e3)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        e2 = self.Att3(g=d3, x=e2)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        e1 = self.Att2(g=d2, x=e1)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        out = self.Conv(d2)

        #  out = self.active(out)

        return out


# For nested 3 channels are required

class conv_block_nested(nn.Module):

    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)

        return output


# Nested Unet

class NestedUNet(nn.Module):
    """
    Implementation of this paper:
    https://arxiv.org/pdf/1807.10165.pdf
    """

    def __init__(self, args):
        super(NestedUNet, self).__init__()
        in_ch = args['in_channels']
        num_classes = args['num_classes']
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0] * 2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1] * 2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2] * 2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0] * 3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1] * 3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0] * 4 + filters[1], filters[0], filters[0])

        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up(x1_3)], 1))

        output = self.final(x0_4)
        return output


# Dictioary Unet
# if required for getting the filters and model parameters for each step

class ConvolutionBlock(nn.Module):
    """Convolution block"""

    def __init__(self, in_filters, out_filters, kernel_size=3, batchnorm=True, last_active=F.relu):
        super(ConvolutionBlock, self).__init__()

        self.bn = batchnorm
        self.last_active = last_active
        self.c1 = nn.Conv2d(in_filters, out_filters, kernel_size, padding=1)
        self.b1 = nn.BatchNorm2d(out_filters)
        self.c2 = nn.Conv2d(out_filters, out_filters, kernel_size, padding=1)
        self.b2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        x = self.c1(x)
        if self.bn:
            x = self.b1(x)
        x = F.relu(x)
        x = self.c2(x)
        if self.bn:
            x = self.b2(x)
        x = self.last_active(x)
        return x


class ContractiveBlock(nn.Module):
    """Deconvuling Block"""

    def __init__(self, in_filters, out_filters, conv_kern=3, pool_kern=2, dropout=0.5, batchnorm=True):
        super(ContractiveBlock, self).__init__()
        self.c1 = ConvolutionBlock(in_filters=in_filters, out_filters=out_filters, kernel_size=conv_kern,
                                   batchnorm=batchnorm)
        self.p1 = nn.MaxPool2d(kernel_size=pool_kern, ceil_mode=True)
        self.d1 = nn.Dropout2d(dropout)

    def forward(self, x):
        c = self.c1(x)
        return c, self.d1(self.p1(c))


class ExpansiveBlock(nn.Module):
    """Upconvole Block"""

    def __init__(self, in_filters1, in_filters2, out_filters, tr_kern=3, conv_kern=3, stride=2, dropout=0.5):
        super(ExpansiveBlock, self).__init__()
        self.t1 = nn.ConvTranspose2d(in_filters1, out_filters, tr_kern, stride=2, padding=1, output_padding=1)
        self.d1 = nn.Dropout(dropout)
        self.c1 = ConvolutionBlock(out_filters + in_filters2, out_filters, conv_kern)

    def forward(self, x, contractive_x):
        x_ups = self.t1(x)
        x_concat = torch.cat([x_ups, contractive_x], 1)
        x_fin = self.c1(self.d1(x_concat))
        return x_fin


class Unet_dict(nn.Module):
    """Unet which operates with filters dictionary values"""

    def __init__(self, n_labels, n_filters=32, p_dropout=0.5, batchnorm=True):
        super(Unet_dict, self).__init__()
        filters_dict = {}
        filt_pair = [3, n_filters]

        for i in range(4):
            self.add_module('contractive_' + str(i), ContractiveBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm))
            filters_dict['contractive_' + str(i)] = (filt_pair[0], filt_pair[1])
            filt_pair[0] = filt_pair[1]
            filt_pair[1] = filt_pair[1] * 2

        self.bottleneck = ConvolutionBlock(filt_pair[0], filt_pair[1], batchnorm=batchnorm)
        filters_dict['bottleneck'] = (filt_pair[0], filt_pair[1])

        for i in reversed(range(4)):
            self.add_module('expansive_' + str(i),
                            ExpansiveBlock(filt_pair[1], filters_dict['contractive_' + str(i)][1], filt_pair[0]))
            filters_dict['expansive_' + str(i)] = (filt_pair[1], filt_pair[0])
            filt_pair[1] = filt_pair[0]
            filt_pair[0] = filt_pair[0] // 2

        self.output = nn.Conv2d(filt_pair[1], n_labels, kernel_size=1)
        filters_dict['output'] = (filt_pair[1], n_labels)
        self.filters_dict = filters_dict

    # final_forward
    def forward(self, x):
        c00, c0 = self.contractive_0(x)
        c11, c1 = self.contractive_1(c0)
        c22, c2 = self.contractive_2(c1)
        c33, c3 = self.contractive_3(c2)
        bottle = self.bottleneck(c3)
        u3 = F.relu(self.expansive_3(bottle, c33))
        u2 = F.relu(self.expansive_2(u3, c22))
        u1 = F.relu(self.expansive_1(u2, c11))
        u0 = F.relu(self.expansive_0(u1, c00))
        return F.softmax(self.output(u0), dim=1)

# Need to check why this Unet is not workin properly
#
# class Convolution2(nn.Module):
#     """Convolution Block using 2 Conv2D
#     Args:
#         in_channels = Input Channels
#         out_channels = Output Channels
#         kernal_size = 3
#         activation = Relu
#         batchnorm = True
#
#     Output:
#         Sequential Relu output """
#
#     def __init__(self, in_channels, out_channels, kernal_size=3, activation='Relu', batchnorm=True):
#         super(Convolution2, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernal_size = kernal_size
#         self.batchnorm1 = batchnorm
#
#         self.batchnorm2 = batchnorm
#         self.activation = activation
#
#         self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, self.kernal_size,  padding=1, bias=True)
#         self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, self.kernal_size, padding=1, bias=True)
#
#         self.b1 = nn.BatchNorm2d(out_channels)
#         self.b2 = nn.BatchNorm2d(out_channels)
#
#         if self.activation == 'LRelu':
#             self.a1 = nn.LeakyReLU(inplace=True)
#         if self.activation == 'Relu':
#             self.a1 = nn.ReLU(inplace=True)
#
#         if self.activation == 'LRelu':
#             self.a2 = nn.LeakyReLU(inplace=True)
#         if self.activation == 'Relu':
#             self.a2 = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         x1 = self.conv1(x)
#
#         if self.batchnorm1:
#             x1 = self.b1(x1)
#
#         x1 = self.a1(x1)
#
#         x1 = self.conv2(x1)
#
#         if self.batchnorm2:
#             x1 = self.b1(x1)
#
#         x = self.a2(x1)
#
#         return x
#
#
# class UNet(nn.Module):
#     """Implementation of U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)
#         https://arxiv.org/abs/1505.04597
#         Args:
#             n_class = no. of classes"""
#
#     def __init__(self, n_class, dropout=0.4):
#         super(UNet, self).__init__()
#
#         in_ch = 3
#         n1 = 64
#         n2 = n1*2
#         n3 = n2*2
#         n4 = n3*2
#         n5 = n4*2
#
#         self.dconv_down1 = Convolution2(in_ch, n1)
#         self.dconv_down2 = Convolution2(n1, n2)
#         self.dconv_down3 = Convolution2(n2, n3)
#         self.dconv_down4 = Convolution2(n3, n4)
#         self.dconv_down5 = Convolution2(n4, n5)
#
#         self.maxpool1 = nn.MaxPool2d(2)
#         self.maxpool2 = nn.MaxPool2d(2)
#         self.maxpool3 = nn.MaxPool2d(2)
#         self.maxpool4 = nn.MaxPool2d(2)
#
#         self.upsample1 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample2 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample3 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#         self.upsample4 = nn.Upsample(scale_factor=2)#, mode='bilinear', align_corners=True)
#
#         self.dropout1 = nn.Dropout(dropout)
#         self.dropout2 = nn.Dropout(dropout)
#         self.dropout3 = nn.Dropout(dropout)
#         self.dropout4 = nn.Dropout(dropout)
#         self.dropout5 = nn.Dropout(dropout)
#         self.dropout6 = nn.Dropout(dropout)
#         self.dropout7 = nn.Dropout(dropout)
#         self.dropout8 = nn.Dropout(dropout)
#
#         self.dconv_up4 = Convolution2(n4 + n5, n4)
#         self.dconv_up3 = Convolution2(n3 + n4, n3)
#         self.dconv_up2 = Convolution2(n2 + n3, n2)
#         self.dconv_up1 = Convolution2(n1 + n2, n1)
#
#         self.conv_last = nn.Conv2d(n1, n_class, kernel_size=1, stride=1, padding=0)
#       #  self.active = torch.nn.Sigmoid()
#
#
#
#     def forward(self, x):
#         conv1 = self.dconv_down1(x)
#         x = self.maxpool1(conv1)
#        # x = self.dropout1(x)
#
#         conv2 = self.dconv_down2(x)
#         x = self.maxpool2(conv2)
#        # x = self.dropout2(x)
#
#         conv3 = self.dconv_down3(x)
#         x = self.maxpool3(conv3)
#        # x = self.dropout3(x)
#
#         conv4 = self.dconv_down4(x)
#         x = self.maxpool4(conv4)
#         #x = self.dropout4(x)
#
#         x = self.dconv_down5(x)
#
#         x = self.upsample4(x)
#         x = torch.cat((x, conv4), dim=1)
#         #x = self.dropout5(x)
#
#         x = self.dconv_up4(x)
#         x = self.upsample3(x)
#         x = torch.cat((x, conv3), dim=1)
#        # x = self.dropout6(x)
#
#         x = self.dconv_up3(x)
#         x = self.upsample2(x)
#         x = torch.cat((x, conv2), dim=1)
#         #x = self.dropout7(x)
#
#         x = self.dconv_up2(x)
#         x = self.upsample1(x)
#         x = torch.cat((x, conv1), dim=1)
#         #x = self.dropout8(x)
#
#         x = self.dconv_up1(x)
#
#         x = self.conv_last(x)
#      #   out = self.active(x)
#
#         return x
