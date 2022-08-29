# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, \
    generate_binary_structure
import torch


# class runningScore(object):
#     def __init__(self, n_classes):
#         self.n_classes = n_classes
#         self.confusion_matrix = np.zeros((n_classes, n_classes))
#
#     def _fast_hist(self, label_true, label_pred, n_class):
#         mask = (label_true >= 0) & (label_true < n_class)
#         hist = np.bincount(n_class*label_true[mask].astype(int)+label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
#         return hist
#
#     def update(self, label_trues, label_preds):
#         for lt, lp in zip(label_trues, label_preds):
#             self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
#
#     def get_scores(self):
#         hist = self.confusion_matrix
#         acc = np.diag(hist).sum() / hist.sum()
#         acc_cls = np.diag(hist) / hist.sum(axis=1)
#         acc_cls = np.nanmean(acc_cls)
#         iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
#         dice=np.divide(np.multiply(iu,2),np.add(iu,1))
#         mean_iu = np.nanmean(iu[1:9])
#         mean_dice=(mean_iu*2)/(mean_iu+1)
#         freq = hist.sum(axis=1) / hist.sum()
#         fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
#         cls_iu = dict(zip(range(self.n_classes), iu))
#
#         return {#'Overall Acc: \t': acc,
#                 #'Mean Acc : \t': acc_cls,
#                 #'FreqW Acc : \t': fwavacc,
#                 'Dice : \t': dice,
#                 'Mean Dice : \t': mean_dice,}, cls_iu
#     def reset(self):
#         self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred):
        n_classes = self.n_classes
        mask = (label_true >= 0) & (label_true < n_classes)
        # print('mask:', mask.shape)
        # print('label_true:', label_true.shape)
        # print('label_pred:', label_pred.shape)
        hist = np.bincount(n_classes * label_true[mask].astype(int) + label_pred[mask],
                           minlength=n_classes ** 2).reshape(n_classes, n_classes)
        return hist

    def update(self, label_trues, label_preds):
        self.confusion_matrix += self._fast_hist(label_trues.flatten(), label_preds.flatten())
        # for lt, lp in zip(label_trues, label_preds):
        #     self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def _calc_iu(self):
        iu = np.zeros(self.n_classes)
        hist = self.confusion_matrix
        temp1 = np.diag(hist)
        temp2 = (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        for ind in range(temp2.size):
            if temp2[ind] == 0:
                iu[ind] = 0
            else:
                iu[ind] = temp1[ind] / temp2[ind]
        return iu

    def get_scores(self):
        #         hist = self.confusion_matrix
        iu = self._calc_iu()
        dice = np.divide(np.multiply(iu, 2), np.add(iu, 1))
        mean_iu = np.nanmean(iu[1:])
        mean_dice = (mean_iu * 2) / (mean_iu + 1)
        return {'Dice': dice, 'Mean Dice': mean_dice, }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def dc(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    intersection = np.count_nonzero(result & reference)

    size_i1 = np.count_nonzero(result)
    size_i2 = np.count_nonzero(reference)

    try:
        dc = 2. * intersection / float(size_i1 + size_i2)
    except ZeroDivisionError:
        dc = 0.0

    return dc


def mhd(result, reference):
    hd1 = __surface_distances(result, reference).mean()
    hd2 = __surface_distances(reference, result).mean()
    hd = max(hd1, hd2)
    return hd

def hd(predictions, labels):
    """Calculates the categorical Dice similarity coefficients for each class
        between labels and predictions.
    Args:
        predictions (np.ndarray): predictions
        labels (np.ndarray): labels
        num_classes (int): number of classes to calculate the dice
            coefficient for
    Returns:
        np.ndarray: dice coefficient per class
    """

    import SimpleITK as sitk
    p = sitk.GetImageFromArray(predictions.astype(np.uint8))
    l = sitk.GetImageFromArray(labels.astype(np.uint8))
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

    hausdorff_distance_filter.Execute(p, l)

    hd_value = hausdorff_distance_filter.GetHausdorffDistance()

    return hd_value 

def hd95(result, reference):
    hd1 = __surface_distances(result, reference)
    hd2 = __surface_distances(reference, result)
#    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    hd95 = np.sort(np.hstack((hd1, hd2)))
    print(hd95[-200])
    return hd95[-200]
#    return hd95 

def __surface_distances(result, reference):
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))

    # binary structure
    footprint = generate_binary_structure(result.ndim, 1)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=None)
    sds = dt[result_border]

    return sds


def metrics(pred, label):
    pred = torch.stack(pred, dim=0).unsqueeze(dim=0)
    label = torch.cat(label, dim=0).unsqueeze(dim=0)
    num_class = label.max() + 1
    print(num_class)
    l, r, c = pred.shape[1:]
    pred_n = torch.zeros([num_class, l, r, c]).scatter_(0, pred, 1)
    label_n = torch.zeros([num_class, l, r, c]).scatter_(0, label, 1)
    pred_n = pred_n.numpy()
    label_n = label_n.numpy()
    dice = []
    hausdorff_distance = []
    for i in range(1, num_class):
        dice.append(dc(pred_n[i, :, :, :], label_n[i, :, :, :]))
        hausdorff_distance.append(mhd(pred_n[i, :, :, :], label_n[i, :, :, :]))
    #hausdorff_distance = hd(pred_n, label_n, num_class)
    score = {'Dice': dice, 'MHD': hausdorff_distance}

    return score
