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
         
                                           
        return seg_final_out, seg_body_out, seg_edge_out



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
        h_grid = torch.linspace(-1.0, 1.0, out_h).view(-1, 1).repeat(1, out_w) 
        w_gird = torch.linspace(-1.0, 1.0, out_w).repeat(out_h, 1)
        grid = torch.cat((w_gird.unsqueeze(2), h_grid.unsqueeze(2)), 2)
        grid = grid.repeat(n, 1, 1, 1).type_as(input).to(input.device)
        
        grid = grid + flow.permute(0, 2, 3, 1) / norm  
       
        output = F.grid_sample(input, grid) 
        return output



