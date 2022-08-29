from preprocess import MICCAI_Img, MICCAI_Label, get_mask_region, MICCAI
from models import NestedUNet, U_Net,shape_Unet,edge_Net,supv_UNet,decouple_Net,Decouple,decouple_body,decouple_multiscale,decouple_res
from AttU_Net import AttU_Net
import numpy  as np
import torch
import torch.nn as nn
from torch.utils import data
import torch.nn.functional as F

import time
import os
from tqdm import tqdm

from utils import adjust_learning_rate
from loss import dice_loss
from metrics import runningScore
from scipy.ndimage.morphology import distance_transform_edt
from utils import adjust_learning_rate, elastic_deformation

def preprocess(PATH, IDs_train, IDs_val, label_s, label_t, is_flip=False, is_rotate=False, is_rgb=False):
    # train
    train_img = MICCAI_Img()
    train_img.read(PATH, IDs_train)

    train_label = MICCAI_Label()
    train_label.read(PATH, IDs_train)
    train_label.trans_vol_label(label_s, label_t)

    mask_region = get_mask_region(train_label.vols, 16, 1)

    train_img.crop(mask_region)
    train_label.crop(mask_region)

    train_img.histeq()

    if is_flip:
        train_img.flip()
        train_label.flip()

    if is_rotate:
        train_img.rotate()
        train_label.rotate()

    if is_rgb:
        train_img.RGB()

    train_img.split()
    train_label.split()
    train_img.transform()
    train_label.transform()

    # val
    val_img = MICCAI_Img()
    val_img.read(PATH, IDs_val)

    val_label = MICCAI_Label()
    val_label.read(PATH, IDs_val)

    val_img.crop(mask_region)
    val_label.crop(mask_region)
    val_label.trans_vol_label(label_s, label_t)

    val_img.histeq()

    if is_rgb:
        val_img.RGB()

    val_img.split()
    val_label.split()

    val_img.transform()
    val_label.transform()
    train_dataset = MICCAI(train_img.vols, train_label.vols)
    val_dataset = MICCAI(val_img.vols, val_label.vols)

    return train_dataset, val_dataset

def mask_to_onehot(mask, num_classes=13):
    _mask = [mask == i for i in range(1, num_classes+1)]
    _mask = [np.expand_dims(x, 0) for x in _mask]
    return np.concatenate(_mask, 0)

def onehot_to_binary_edges(mask, radius=1, num_classes=13):
    if radius < 0:
        return mask

    # We need to pad the borders for boundary conditions
    mask_pad = np.pad(mask, ((0, 0),(1, 1), (1, 1)), mode='constant', constant_values=0)

    edgemap = np.zeros(mask.shape[1:])

    for i in range(num_classes):
        dist = distance_transform_edt(mask_pad[i, :])+distance_transform_edt(1.0-mask_pad[i, :])
        dist = dist[1:-1, 1:-1]
        dist[dist > radius] = 0
        edgemap += dist
    edgemap = np.expand_dims(edgemap, axis=0)
    edgemap = (edgemap > 0).astype(np.uint8)
    return edgemap

def mask_to_edges(mask):
    _edge = mask
    _edge = mask_to_onehot(_edge)
    _edge = onehot_to_binary_edges(_edge)
    return torch.from_numpy(_edge).float()

def train(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args['gpu_id'])
    PATH = args['PATH']
    IDs_train = args['IDs_train']
    IDs_val = args['IDs_val']
    label_source = args['label_source']
    label_target = args['label_target']
    model_args = args['model_args']
    n_epoch = args['n_epoch']
    experiment_i = args['experiment_i']
    path_save = args['path_save']
    t_begin = time.time()

    train_dataset, val_dataset = preprocess(PATH, IDs_train, IDs_val, label_source, label_target, is_flip=False,
                                            is_rotate=True, is_rgb=False)
    train_loader = data.DataLoader(train_dataset, batch_size=8, num_workers=1, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=True)

    model = decouple_Net(model_args)
    model = model.cuda()

    loss_ce = F.cross_entropy
    loss_dc = dice_loss
    loss_sr = nn.L1Loss(reduction='mean') #SR_loss
    loss_edge = nn.BCELoss(reduction='mean')
    init_lr = 1e-3
    #optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8)
    num_classes = model_args['num_classes']
    running_metrics = runningScore(num_classes)
    score_dice = []
    score_dice_best = 0
    max_score = {}
    max_dice = 0
    print('--------------------------------Training--------------------------------')
    for epoch in range(n_epoch):
        print('Experiment:', experiment_i)
        print('epoch: ', epoch + 1)
        model.train()
        adjust_learning_rate(init_lr, optimizer, epoch)
        loss_epoch = 0.0
        t_epoch = time.time()
        for img, label in tqdm(train_loader):
            img = img.cuda()
            label = label.cuda()
            label_ = label.cpu() #8, 176, 160         
            label_ = label[0, :, :].cpu() #176, 160 
            edge = mask_to_edges(label_).cuda()
            
            optimizer.zero_grad()
            #outputs = model(img)   
            #outputs,edge_out = model(img)
            outputs,body_out, edge_out = model(img) #
            #outputs,o4,o3,o2,o1 = model(img) 
            #outputs,edge_out,o4,o3,o2,o1 = model(img) #8, 8, 176, 160; 8, 1, 176, 160 ,o4,o3,o2,o1
            edge_out = edge_out[0, :, :, :]
            #loss = loss_ce(outputs, label) + loss_dc(outputs=outputs, gt=label) +loss_edge(edge_out, edge)
            loss = loss_ce(outputs, label) + loss_dc(outputs=outputs, gt=label) + loss_ce(body_out, label)+ loss_dc(outputs=body_out, gt=label)+loss_edge(edge_out, edge)#+loss_de(outputs=edge_out, gt=edge.long()) 
            #loss = 0.5 * loss_ce(outputs, label) + 0.5 * loss_dc(outputs=outputs, gt=label)
            #loss = loss_ce(outputs, label)+ loss_ce(o1, label)+loss_ce(o2, label)+ loss_ce(o3, label)+ loss_ce(o4, label) #

            loss.backward()
            optimizer.step()
            loss_epoch += loss.item()
            model.train()

        loss_epoch /= len(train_loader)
        t_train = time.time()

        print('average loss in this epoch: ', loss_epoch)
        print('final loss in this epoch: ', loss.data.item())
        print('cost {} seconds up to now'.format(t_train - t_begin))
        print('cost {} seconds in this train epoch'.format(t_train - t_epoch))

        model.eval()

        for i_val, (img, label) in enumerate(val_loader):
            img = img.cuda()
            label = label.numpy()
            with torch.no_grad():
                outputs, _, _ = model(img)#, _, _
                outputs = outputs[0, :, :, :]
            pred = outputs.data.max(0)[1].cpu().numpy()
            running_metrics.update(label, pred)
            score = running_metrics.get_scores()

        print('Mean Dice: ', score['Mean Dice'])
        print('Left_Thalamus: ', score['Dice'][1])
        print('Left_Caudate: ', score['Dice'][2])
        print('Left_Putamen: ', score['Dice'][3])
        print('Left_Pallidum: ', score['Dice'][4])
        print('Left_Hippocampus: ', score['Dice'][5])
        print('Left_Amygdala: ', score['Dice'][6])
        print('Right_Thalamus: ', score['Dice'][7])
        print('Right_Caudate: ', score['Dice'][8])
        print('Right_Putamen: ', score['Dice'][9])
        print('Right_Pallidum: ', score['Dice'][10])
        print('Right_Hippocampus: ', score['Dice'][11])
        print('Right_Amygdala: ', score['Dice'][12])
        print('----------------------------------------------------------------')
        score_dice.append(score['Mean Dice'])
               
        if score_dice[epoch] > score_dice_best:  
            score_dice_best = score_dice[epoch]  
            dict_name = os.path.join(path_save, str(experiment_i) + '_best_epoch' + '.pkl')
            torch.save(model.state_dict(), dict_name) 
            score_best = score

        running_metrics.reset() 

    if args['save_dict']:
        path_save = args['path_save']
        dict_name = os.path.join(path_save, str(experiment_i) + '.pkl')
        torch.save(model.state_dict(), dict_name)

    return score_best


if __name__ == '__main__':
    cur_path = os.getcwd()
    path = os.path.join(cur_path, 'MICCAI')
    args = dict(PATH=path, gpu_id=1)


    args['label_source'] = (59, 60, 36, 37, 57, 58, 55, 56, 47, 48, 31, 32)
    args['label_target'] = (1,2,3,4,5,6,7,8,9,10,11,12)
    #args['label_target'] = (1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6)
    args['n_epoch'] = 15
    args['model_args'] = {'in_channels': 1, 'num_classes': 13}
    args['experiment_i'] = 10
    args['save_dict'] = True

    if args['save_dict']:
        localtime = time.asctime(time.localtime(time.time()))
        localtime = localtime.replace(' ', '_')
        localtime = localtime.replace(':', '_')
        print(localtime)
        path_save = os.path.join(cur_path, localtime)
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        args['path_save'] = path_save
    IDs = [(['12', '26', '34', '02', '24', '17', '03', '15', '18', '01', '22', '23', '27', '05', '30'],
          ['06', '32', '09', '13', '29', '25', '31', '19', '07', '20', '14', '10', '35', '08', '21', '33', '28', '11', '16', '04']),
        (['07', '27', '26', '09', '08', '24', '35', '34', '12', '04', '16', '11', '01', '22', '23'],
         ['06', '32', '02', '13', '29', '25', '31', '15', '19', '17', '18', '20', '14', '10', '21', '33', '28', '30', '05', '03']),
        (['34', '35', '20', '23', '27', '17', '31', '12', '14', '09', '03', '08', '04', '01', '05'],
         ['06', '32', '02', '13', '29', '25', '15', '19', '07', '18', '26', '22', '10', '21', '33', '24', '28', '30', '11', '16']),
        (['12', '24', '27', '19', '13', '26', '15', '34', '06', '21', '30', '33', '35', '17', '08'],
         ['32', '02', '09', '29', '25', '31', '07', '23', '18', '01', '20', '22', '14', '10', '28', '11', '05', '16', '04', '03']),
        (['07', '12', '14', '06', '11', '35', '19', '16', '24', '23', '25', '34', '21', '26', '22'],
         ['32', '27', '02', '09', '13', '29', '31', '15', '17', '18', '01', '20', '10', '08', '33', '28', '30', '05', '04', '03']),
        (['33', '06', '28', '10', '13', '15', '18', '27', '12', '03', '02', '19', '14', '16', '23'],
         ['26', '22', '32', '35', '09', '08', '29', '21', '25', '31', '24', '30', '11', '05', '07', '04', '34', '17', '01', '20']),
        (['12', '21', '11', '29', '35', '25', '31', '16', '15', '06', '27', '08', '32', '03', '07'],
         ['26', '22', '14', '10', '02', '09', '13', '33', '24', '28', '30', '18', '05', '19', '04', '23', '34', '17', '01', '20']),
        (['33', '24', '12', '11', '31', '13', '27', '15', '04', '08', '23', '29', '03', '09', '17'],
         ['26', '22', '06', '32', '14', '10', '35', '02', '21', '28', '25', '30', '18', '05', '19', '07', '16', '34', '01', '20']),
        (['32', '21', '15', '35', '34', '26', '11', '09', '30', '22', '03', '20', '06', '18', '10'],
         ['27', '02', '13', '29', '25', '31', '19', '07', '23', '17', '01', '14', '08', '33', '24', '28', '12', '05', '16', '04']),
        (['19', '02', '35', '16', '21', '03', '31', '34', '23', '24', '32', '17', '28', '07', '15'],
         ['06', '27', '09', '13', '29', '25', '18', '01', '20', '26', '22', '14', '10', '08', '33', '12', '30', '11', '05', '04']),
        (['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15'],
         ['16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35'])]

    args['IDs_train'], args['IDs_val'] = IDs[args['experiment_i']]
    train(args)
