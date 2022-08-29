import torch
import torch.nn.functional as F
import numpy as np
from numpy import ma
import os


def cross_entropy2d(outputs, gt, weight=None, size_average=True):
    n, c, h, w = outputs.size()
    nt, ct, ht, wt = gt.size()
    '''
    # Handle inconsistent size between input and target
    if h > ht and w > wt: # upsample labels
        target = target.unsequeeze(1)
        target = F.upsample(target, size=(h, w), mode='nearest')
        target = target.sequeeze(1)
    elif h < ht and w < wt: # upsample images
        input = F.upsample(input, size=(ht, wt), mode='bilinear')
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    '''
    log_p = F.log_softmax(outputs, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[gt.contiguous().view(-1, 1).repeat(1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = gt >= 0
    gt = gt[mask]
    loss = F.nll_loss(log_p, gt, ignore_index=250,
                      weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss


def cross_entropy3d(outputs, gt, weight=None, size_average=True):
    num_class = outputs.shape[1]
    log_p = F.log_softmax(outputs, dim=1)
    log_p = log_p.permute(0, 2, 3, 4, 1).contiguous().view(-1, num_class)
    log_p = log_p[gt.contiguous().view(-1, 1).repeat(1, num_class) >= 0]
    log_p = log_p.view(-1, num_class)

    mask = gt >= 0
    gt = gt[mask]
    loss = F.nll_loss(log_p, gt, ignore_index=250, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum().float()
    return loss


def test_CrossEntropyLoss(outputs, gt, weight=None):
    if outputs.dim() != gt.dim():
        gt = gt.unsqueeze(dim=1)
    gt_bin = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    return F.cross_entropy(outputs, gt_bin, weight=weight)



def dice_loss(outputs, gt):
    if outputs.dim() != gt.dim():
        gt = gt.unsqueeze(dim=1)
    pred = torch.softmax(outputs, dim=1)
    gt_bin = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    # gt_bin = torch.zeros(pred.shape).scatter_(1, gt, 1)
    gt_bin = gt_bin[:, 1:, :, :]
    smooth = 0
    pred = pred[:, 1:, :, :]
    gt_flat = gt_bin.contiguous().view(-1)
    pred_flat = pred.contiguous().view(-1)

    intersection = (gt_flat * pred_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth))

def dice_edge(outputs, gt):
    if outputs.dim() != gt.dim():
        gt = gt.unsqueeze(dim=1)
    pred = torch.softmax(outputs, dim=1)
    gt_bin = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    # gt_bin = torch.zeros(pred.shape).scatter_(1, gt, 1)
    #gt_bin = gt_bin[:, 1:, :, :]
    smooth = 0
    #pred = pred[:, 1:, :, :]
    gt_flat = gt_bin.contiguous().view(-1)
    pred_flat = pred.contiguous().view(-1)

    intersection = (gt_flat * pred_flat).sum()
    return 1 - ((2. * intersection + smooth) / (pred_flat.sum() + gt_flat.sum() + smooth))

def gen_dice_loss(outputs, gt):
    if outputs.dim() != gt.dim():
        gt = gt.unsqueeze(dim=1)
    pred = torch.softmax(outputs, dim=1)
    gt_bin = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    # gt_bin = torch.zeros(pred.shape).scatter_(1, gt, 1)
    gt_bin = gt_bin[:, 1:, :, :]
    smooth = 0
    num_classes = outputs.shape[1] - 1
    pred = pred[:, 1:, :, :]

    prob = pred.contiguous().view(num_classes, -1)
    gt_n = gt_bin.contiguous().view(num_classes, -1)

    w = 1/(gt_n**2+0.000001)

    intersection = prob * gt_n
    loss = (2 * (w * intersection).sum(1) + smooth) / ((w * prob).sum(1) + (w * gt_n).sum(1) + smooth)
    loss = 1 - loss.sum() / num_classes
    return loss


# An implementation of Voxel-Level Hardness-Weighted Dice Loss
def hwd_loss(outputs, gt, l=0.5):
    gt = gt.unsqueeze(dim=1)
    prob = torch.softmax(outputs, dim=1)
    smooth = 0
    num_classes = outputs.shape[1] - 1
    gt_n = torch.zeros(outputs.shape).cuda().scatter_(1, gt, 1)
    prob = prob[:, 1:, :, :]
    gt_n = gt_n[:, 1:, :, :]
    # gt_n = torch.zeros(pred.shape).scatter_(1, gt, 1)
    prob = prob.contiguous().view(num_classes, -1)
    gt_n = gt_n.contiguous().view(num_classes, -1)
    w = l * abs(prob - gt_n) + 1.0 - l
    intersection = prob * gt_n
    loss = (2 * (w * intersection).sum(1) + smooth) / ((w * prob).sum(1) + (w * gt_n).sum(1) + smooth)
    loss = 1 - loss.sum() / num_classes
    return loss


