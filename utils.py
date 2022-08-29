import torch
from torch import nn
from torch.nn import init

import numpy as np
import cv2 as cv
import os.path
import nibabel as nib
import math
from PIL import Image
from elasticdeform import deform_random_grid
from skimage import util

def to_uint8(vol):
    vol = vol.astype(np.float)
    vol[vol < 0] = 0
    return ((vol - vol.min()) * 255.0 / vol.max()).astype(np.uint8)


def IR_to_uint8(vol):
    vol = vol.astype(np.float)
    vol[vol < 0] = 0
    return ((vol - 800) * 255.0 / vol.max()).astype(np.uint8)
    


def histeq_vol(vol):
    for ind in range(vol.shape[0]):
        vol[ind, :, :] = cv.equalizeHist(vol[ind, :, :])
    return vol
    
def elastic_deformation(b_img, b_label):
    imgs = b_img.get_vols()
    labels = b_label.get_vols()
    num_classes = b_label.num_classes
    for ind in range(len(imgs)):
        img, label = imgs[ind], labels[ind]
        img_deformed, label_deformed = deform_random_grid([img, label], sigma=10, points=3, axis=(1, 2))
        label_deformed[label_deformed>(num_classes-1)] = num_classes-1
        label_deformed[label_deformed<0] = 0
        b_img.vols.append(img_deformed)
        b_label.vols.append(label_deformed)
    return b_img, b_label

def read_vol(PATH):
    return nib.load(PATH).get_data().transpose(2, 0, 1)

def random_img_elastic_deformation(img_vols, percentage='20%'): #percentage选中切片中加噪的图像占比
    #img_vols = img.get_vols()
    percent = {'1%': 2.575829, '5%': 1.95964, '10%': 1.644854, '20%': 1.281552, '30%': 1, '50%': 0.674490}
    p = percent[percentage] #p 为 对应的值 
    print(p)
    randnumber = torch.randn(len(img_vols))#生成一组数量为切片数目的 标准正态分布 
    print(randnumber)
    ind = randnumber.gt(p) | randnumber.le(-p) #ind为 >p或 <=-p 
    print(ind.sum()/len(img_vols))#检查均值是否为0
    for num, i in enumerate(ind):
        if i:
            img = img_vols[num]
            #img = deform_random_grid(img, sigma=5, points=3)#加噪程序从标准偏差sigma 的正态分布 采样 
            img = util.random_noise(img, mode='gaussian') #gaussian s&p speckle
            img_vols[num] = img
    return img_vols 
    

def flip_vol(vol):
    flipped_vol = np.zeros(vol.shape)
    for ind in range(vol.shape[0]):
        flipped_vol[ind, :, :] = cv.flip(vol[ind, :, :], 1)
    return flipped_vol


def rotate_vol(vol, angle, interp=cv.INTER_NEAREST):
    rows, cols = vol.shape[1], vol.shape[2]
    rotated_vol = np.zeros(vol.shape)
    M = cv.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
    for ind in range(vol.shape[0]):
        rotated_vol[ind, :, :] = cv.warpAffine(vol[ind, :, :], M, (cols, rows), flags=interp)
    return rotated_vol


def edge_vol(vol, kernel_size=(3, 3), sigmaX=0):
    edge = np.zeros(vol.shape, np.uint8)
    for ind in range(vol.shape[0]):
        edge[ind, :, :] = cv.Canny(vol[ind, :, :], 1, 1)
        edge[ind, :, :] = cv.GaussianBlur(edge[ind, :, :], kernel_size, sigmaX)
    return edge


def split_vols(vols):
    temp = []
    for vol in vols:
        temp.extend(np.split(vol, vol.shape[0], 0))

    return [i.squeeze(axis=0) for i in temp]


def stack_vol(vol, stack_num):
    assert stack_num % 2 == 1, 'stack numbers must be odd!'
    vol = np.expand_dims(vol, axis=1)
    N = range(stack_num // 2, -(stack_num // 2 + 1), -1)
    stacked_vol = np.roll(vol, N[0], axis=0)
    for n in N[1:]:
        stacked_vol = np.concatenate((stacked_vol, np.roll(vol, n, axis=0)), axis=1)
    return stacked_vol


# crop
def get_vol_region(mask, scale1, scale2):
    l, r, c = np.where(mask > 0)
    min_l, min_r, min_c = l.min(), r.min(), c.min()
    max_l, max_r, max_c = l.max(), r.max(), c.max()
    max_r = min_r + calc_ceil_pad(max_r - min_r, scale1)
    max_c = min_c + calc_ceil_pad(max_c - min_c, scale1)
    max_l = min_l + calc_ceil_pad(max_l - min_l, scale2)

    return [(min_l, max_l), (min_r - scale1, max_r + scale1), (min_c - scale1, max_c + scale1)]


def calc_ceil_pad(x, devider):
    return math.ceil(x / float(devider)) * devider


def crop_vol(vol, crop_region):
    l_range, r_range, c_range = crop_region
    cropped_vol = vol[l_range[0]: l_range[1], r_range[0]: r_range[1],
                  c_range[0]: c_range[1]]
    return cropped_vol


def get_mask_region(vols, scale1, scale2):
    mask = np.zeros(vols[0].shape[:])
    for vol in vols:
        l, r, c = np.where(vol > 0)
        mask[l, r, c] = 1
    mask_region = get_vol_region(mask, scale1, scale2)
    return mask_region


class BrainData:
    def __init__(self):
        self.vols = []

    def get_vols(self):
        return self.vols

    def read(self, PATH, IDs, suffix):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_vol(os.path.join(PATH, id_vol + '_' + suffix))))

    def histeq(self):
        self.vols = [histeq_vol(vol) for vol in self.vols]

    def flip(self):
        for ind in range(len(self.vols)):
            self.vols.append(flip_vol(self.vols[ind]))

    def rotate(self, angle_list=None):
        if angle_list is None:
            angle_list = [5, -5, 10, -10, 15, -15]
        length = len(self.vols)
        for angle in angle_list:
            for ind in range(length):
                self.vols.append(rotate_vol(self.vols[ind], angle))

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region[ind])
            self.vols[ind] = cropped_vol

    def edge(self):
        assert self.modality == 'label', 'must be label'
        self.edge = [edge_vol(vol) for vol in self.vols]
    
    #def normalize(self):
        #self.vols = [(vol - np.mean(vol))/np.std(vol) for vol in self.vols]
    #def transform(self):
        #if self.vols[0].ndim == 2:
            #self.vols = [torch.from_numpy((np.expand_dims(vol, 0).astype(np.float))).float()
                         #for vol in self.vols]
        #else:
            #self.vols = [torch.from_numpy((vol.astype(np.float))).float()
                         #for vol in self.vols]
                         
    def transform(self):
        mean = [np.mean(vol) for vol in self.vols]
        mean = np.mean(mean)
        if self.vols[0].ndim == 2:
            self.vols = [torch.from_numpy((np.expand_dims(vol, 0).astype(np.float) - mean) / 255.0).float()
                         for vol in self.vols]
        else:
            self.vols = [torch.from_numpy((vol.astype(np.float) - mean) / 255.0).float()
                         for vol in self.vols]
                         
        # if is_3D:
        #     self.vols = [torch.from_numpy((np.expand_dims(vol, 0).astype(np.float) - mean) / 255.0).float()
        #                  for vol in self.vols]
        # else:
        #     self.vols = [torch.from_numpy((vol.astype(np.float) - mean) / 255.0).float()
        #                  for vol in self.vols]
        # self.vols = [torch.from_numpy((np.expand_dims(vol, 0).astype(np.float) - mean) / 255.0).float()
        #              for vol in self.vols]

    def stack(self, stack_num):
        self.vols = [stack_vol(vol, stack_num) for vol in self.vols]
        # self.vols = split_vols(vols)

    def split(self):
        self.vols = split_vols(self.vols)


def model_size(model, input, type_size=4):
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * type_size / 1000 / 1000))

    input_ = input.clone()
    input_.requires_grad_(requires_grad=False)

    mods = list(model.modules())
    out_sizes = []

    for i in range(1, len(mods)):
        m = mods[i]
        if isinstance(m, nn.ReLU):
            if m.inplace:
                continue
        out = m(input_)
        out_sizes.append(np.array(out.size()))
        input_ = out

    total_nums = 0
    for i in range(len(out_sizes)):
        s = out_sizes[i]
        nums = np.prod(np.array(s))
        total_nums += nums

    print('Model {} : intermedite variables: {:3f} M (without backward)'
          .format(model._get_name(), total_nums * type_size / 1000 / 1000))
    print('Model {} : intermedite variables: {:3f} M (with backward)'
          .format(model._get_name(), total_nums * type_size * 2 / 1000 / 1000))


### compute model params
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def adjust_learning_rate(init_lr, optimizer, epoch):
    lr = init_lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


### initalize the module
def init_weights(net, init_type='normal'):
    if init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # print('init_Conv')
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # print('init_BatchNorm')
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)




