import torch
from torch.utils import data
import nibabel as nib
import os.path
import numpy as np
from utils import BrainData, get_mask_region, crop_vol, to_uint8
import copy


def read_IBSR_vol(path):
    vol = nib.load(path).get_data().squeeze().transpose(2, 1, 0)
    return np.flip(vol, axis=2)


class IBSR_Img(BrainData):

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_IBSR_vol(os.path.join(
                PATH, 'IBSR_' + id_vol + '_ana_stripped.nii'))))
        self.affine = nib.load(os.path.join(
                PATH, 'IBSR_' + id_vol + '_ana_stripped.nii')).affine

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def RGB(self):
        for ind in range(len(self.vols)):
            vol_i = self.vols[ind]
            vol_i = np.expand_dims(vol_i, axis=1)
            stacked_vol = np.concatenate((vol_i, vol_i, vol_i), axis=1)
            self.vols[ind] = stacked_vol



class IBSR_Label(BrainData):
    def __int__(self):
        super(IBSR_Label, self).__init__()
        #self.num_class = []

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_IBSR_vol(os.path.join(PATH, 'IBSR_' + id_vol + '_seg.nii')))
        self.affine = nib.load(os.path.join(
                PATH, 'IBSR_' + id_vol + '_seg.nii')).affine 
    def trans_vol_label(self, label_s, label_t):
        assert len(label_s) == len(label_t), 'length must be same!'
        self.num_classes = max(label_t) + 1
        for i in range(len(self.vols)):
            vol = np.zeros(self.vols[i].shape)
            for j in range(len(label_s)):
                l, r, c = np.where(self.vols[i] == label_s[j])
                vol[l, r, c] = label_t[j]
            self.vols[i] = vol

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def transform(self):
        self.vols = [torch.from_numpy(vol.astype(np.long)) for vol in self.vols]


class IBSR_Label_TRI(BrainData):
    def __int__(self):
        super(IBSR_Label_TRI, self).__init__()
        self.num_class = []

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_IBSR_vol(os.path.join(PATH, 'IBSR_' + id_vol + '_segTRI.nii')))

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def transform(self):
        self.vols = [torch.from_numpy(vol.astype(np.long)) for vol in self.vols]


class IBSR(data.Dataset):
    def __init__(self, data):
        self.img = data[0]
        self.label = data[1]

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        return self.img[index], self.label[index]


############################################################################################

def read_MRBrainS_vol(path):
    vol = nib.load(path).get_data().transpose(2, 1, 0)
    return vol


class MRBrainS_img(BrainData):
    def read(self, path, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_IBSR_vol(os.path.join(path, id_vol + '_FLAIR_stripped.nii'))))
            self.vols.append(to_uint8(read_IBSR_vol(os.path.join(path, id_vol + '_reg_IR_stripped.nii'))))
            self.vols.append(to_uint8(read_IBSR_vol(os.path.join(path, id_vol + '_reg_T1_stripped.nii'))))

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol


class MRBrainS_label(BrainData):
    def __init__(self):
        super(MRBrainS_label, self).__init__()
        self.num_class = []

    def read(self, path, IDs):
        for id_vol in IDs:
            self.vols.append(read_IBSR_vol(os.path.join(path, id_vol + '_segm.nii.gz')))

    def trans_vol_label(self, label_s, label_t):
        assert len(label_s) == len(label_t), 'length must be same!'
        self.num_class = max(label_t) + 1
        for i in range(len(self.vols)):
            vol = np.zeros(self.vols[i].shape)
            for j in range(len(label_s)):
                l, r, c = np.where(self.vols[i] == label_s[j])
                vol[l, r, c] = label_t[j]
            self.vols[i] = vol

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def transform(self):
        self.vols = [torch.from_numpy(vol.astype(np.long)) for vol in self.vols]


class MRBrainS(data.Dataset):

    def __init__(self, data):
        self.img0 = data[0]
        self.img1 = data[1]
        self.img2 = data[2]
        self.label = data[3]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.img0[index], self.img1[index], self.img2[index], self.label[index]


##################################################################

def read_LPBA_vol(path):
    vol = nib.load(path).get_data().transpose(1, 2, 0)
    return np.flip(vol, axis=1)


class LPBA_Img(BrainData):

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_LPBA_vol(os.path.join(
                PATH, 'LPBA_' + id_vol + '_img.nii'))))

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def RGB(self):
        for ind in range(len(self.vols)):
            vol_i = self.vols[ind]
            vol_i = np.expand_dims(vol_i, axis=1)
            stacked_vol = np.concatenate((vol_i, vol_i, vol_i), axis=1)
            self.vols[ind] = stacked_vol


class LPBA_Label(BrainData):

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_LPBA_vol(os.path.join(PATH, 'LPBA_' + id_vol + '_seg.nii')))

    def trans_vol_label(self, label_s, label_t):
        assert len(label_s) == len(label_t), 'length must be same!'
        for i in range(len(self.vols)):
            vol = np.zeros(self.vols[i].shape)
            for j in range(len(label_s)):
                l, r, c = np.where(self.vols[i] == label_s[j])
                vol[l, r, c] = label_t[j]
            self.vols[i] = vol

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def transform(self):
        self.vols = [torch.from_numpy(vol.astype(np.long)) for vol in self.vols]


class LPBA(data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        return self.img[index], self.label[index]


##################################################################
def read_MICCAI_vol(path):
    vol = nib.load(path).get_data().transpose(2, 1, 0)
    return vol


class MICCAI_Img(BrainData):

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(to_uint8(read_MICCAI_vol(os.path.join(
                PATH, id_vol + '_img.nii'))))
        self.affine = nib.load(os.path.join(
                PATH, id_vol + '_img.nii')).affine
    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def RGB(self):
        for ind in range(len(self.vols)):
            vol_i = self.vols[ind]
            vol_i = np.expand_dims(vol_i, axis=1)
            stacked_vol = np.concatenate((vol_i, vol_i, vol_i), axis=1)
            self.vols[ind] = stacked_vol


class MICCAI_Label(BrainData):

    def read(self, PATH, IDs):
        for id_vol in IDs:
            self.vols.append(read_MICCAI_vol(os.path.join(PATH, id_vol + '_seg.nii')))
            self.affine = nib.load(os.path.join(
                PATH, id_vol + '_seg.nii')).affine

    def trans_vol_label(self, label_s, label_t):
        assert len(label_s) == len(label_t), 'length must be same!'
        for i in range(len(self.vols)):
            vol = np.zeros(self.vols[i].shape)
            for j in range(len(label_s)):
                l, r, c = np.where(self.vols[i] == label_s[j])
                vol[l, r, c] = label_t[j]
            self.vols[i] = vol

    def crop(self, crop_region):
        for ind in range(len(self.vols)):
            cropped_vol = crop_vol(self.vols[ind], crop_region)
            self.vols[ind] = cropped_vol

    def transform(self):
        self.vols = [torch.from_numpy(vol.astype(np.long)) for vol in self.vols]


class MICCAI(data.Dataset):
    def __init__(self, img, label):
        self.img = img
        self.label = label

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        return self.img[index], self.label[index]


##################################################################
def get_mask(PATH, IDs, label_s, label_t, scale1=16, scale2=1):
    label = IBSR_Label()
    label.read(PATH, IDs)
    label.trans_vol_label(label_s, label_t)
    mask_region = get_mask_region(label.vols, scale1, scale2)
    return mask_region


def preprocess_train(PATH, IDs, label_s, label_t, mask_region,
                     is_histeq=False, is_flip=False, is_rotate=False, is_split=False):
    brain_data_img = IBSR_Img()
    brain_data_img.read(PATH, IDs)

    brain_data_label = IBSR_Label()
    brain_data_label.read(PATH, IDs)
    brain_data_label.trans_vol_label(label_s, label_t)

    brain_data_img.crop(mask_region)
    brain_data_label.crop(mask_region)

    if is_histeq:
        brain_data_img.histeq()

    atlas_img = copy.copy(brain_data_img)
    atlas_img.transform()
    atlas_label = copy.copy(brain_data_label)
    atlas_label.transform()
    atlas_img = [vol.reshape(vol.shape[1], -1) for vol in atlas_img.vols]
    atlas_label = [vol.reshape(vol.shape[0], -1) for vol in atlas_label.vols]

    if is_flip:
        brain_data_img.flip()
        brain_data_label.flip()

    if is_rotate:
        brain_data_img.rotate()
        brain_data_label.rotate()

    if is_split:
        brain_data_img.split()
        brain_data_label.split()
    brain_data_img.normalize()
    brain_data_img.transform()
    brain_data_label.transform()

    return [brain_data_img.vols, brain_data_label.vols], [atlas_img, atlas_label]


def preprocess_val(PATH, IDs, label_s, label_t, mask_region, is_histeq=False, is_split=False):
    brain_data_img = IBSR_Img()
    brain_data_img.read(PATH, IDs)

    brain_data_label = IBSR_Label()
    brain_data_label.read(PATH, IDs)
    brain_data_label.trans_vol_label(label_s, label_t)

    brain_data_img.crop(mask_region)
    brain_data_label.crop(mask_region)

    if is_histeq:
        brain_data_img.histeq()
    brain_data_img.normalize()
    if is_split:
        brain_data_img.split()
        brain_data_label.split()

    brain_data_img.transform()
    brain_data_label.transform()

    return [brain_data_img.vols, brain_data_label.vols]
# def preprocess_train(PATH, IDs, label_s, label_t,
#                      is_histeq=False, is_flip=False, is_rotate=False, is_split=False,
#                      scale1=16, scale2=1):
#     brain_data_img = IBSR_BrainData_Img()
#     brain_data_img.read(PATH, IDs)
#
#     brain_data_label = IBSR_BrainData_Label()
#     brain_data_label.read(PATH, IDs)
#     brain_data_label.trans_vol_label(label_s, label_t)
#
#     mask_region = get_mask_region(brain_data_label.vols, scale1, scale2)
#
#     brain_data_img.crop(mask_region)
#     brain_data_label.crop(mask_region)
#
#     if is_histeq:
#         brain_data_img.histeq()
#
#     if is_flip:
#         brain_data_img.flip()
#         brain_data_label.flip()
#
#     if is_rotate:
#         brain_data_img.rotate()
#         brain_data_label.rotate()
#     if is_split:
#         brain_data_img.split()
#         brain_data_label.split()
#
#     brain_data_img.transform()
#     brain_data_label.transform()
#
#     return brain_data_img, brain_data_label, mask_region
#
#
# def preprocess_val(PATH, IDs, label_s, label_t, mask_region, is_histeq=False, is_split=False):
#     brain_data_img = IBSR_BrainData_Img()
#     brain_data_img.read(PATH, IDs)
#
#     brain_data_label = IBSR_BrainData_Label()
#     brain_data_label.read(PATH, IDs)
#     brain_data_label.trans_vol_label(label_s, label_t)
#
#     brain_data_img.crop(mask_region)
#     brain_data_label.crop(mask_region)
#
#     if is_histeq:
#         brain_data_img.histeq()
#
#     if is_split:
#         brain_data_img.split()
#         brain_data_label.split()
#
#     brain_data_img.transform()
#     brain_data_label.transform()
#
#     return brain_data_img, brain_data_label
