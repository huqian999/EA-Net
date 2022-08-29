import os
import time

import numpy  as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt

from loss import dice_loss, hwd_loss, gen_dice_loss, dice_edge
from metrics import runningScore

from models import decouple_Net
from preprocess import IBSR_Img, IBSR_Label, get_mask_region, IBSR
from utils import adjust_learning_rate, elastic_deformation,random_img_elastic_deformation


def preprocess(PATH, IDs_train, IDs_val, label_s, label_t, is_flip=False, is_rotate=False, is_rgb=False): 
    # train
    train_img = IBSR_Img()
    train_img.read(PATH, IDs_train)

    train_label = IBSR_Label()
    train_label.read(PATH, IDs_train)
    train_label.trans_vol_label(label_s, label_t)

    mask_region = get_mask_region(train_label.vols, 16, 1)

    train_img.crop(mask_region)
    train_label.crop(mask_region)

    train_img.histeq()
    #train_img, train_label = elastic_deformation(train_img, train_label)
    if is_flip:
        train_img.flip()
        train_label.flip()

    if is_rotate:
        train_img.rotate()
        train_label.rotate()

    if is_rgb:
        train_img.RGB()
    #train_img.normalize()
    #train_img.vols = random_img_elastic_deformation(train_img.vols, p)
    
    train_img.split()
    train_label.split()
    train_img.transform()
    train_label.transform()

    # val
    val_img = IBSR_Img()
    val_img.read(PATH, IDs_val)

    val_label = IBSR_Label()
    val_label.read(PATH, IDs_val)

    val_img.crop(mask_region)
    val_label.crop(mask_region)
    val_label.trans_vol_label(label_s, label_t)

    val_img.histeq()

    if is_rgb:
        val_img.RGB()

    val_img.split()
    val_label.split()
    #val_img.normalize()
    val_img.transform()
    val_label.transform()
    train_dataset = IBSR([train_img.vols, train_label.vols])
    val_dataset = IBSR([val_img.vols, val_label.vols])

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
                                            is_rotate=True)
    train_loader = data.DataLoader(train_dataset, batch_size=8, num_workers=1, shuffle=True)
    val_loader = data.DataLoader(val_dataset, batch_size=1, num_workers=1, shuffle=True)


    model = decouple_Net(model_args)#U_Net edge_Net supv_UNet Decouple decouple_Net shape_Unet
    model = model.cuda()

    loss_ce = F.cross_entropy
    loss_dc = dice_loss
    loss_sr = nn.L1Loss(reduction='mean') #SR_loss
    loss_edge = nn.BCELoss(reduction='mean')
    loss_de = dice_edge
    ######################
    init_lr = 1e-3
    ######################
    #optimizer = torch.optim.SGD(model.parameters(), lr=init_lr, momentum=0.99, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-8)
    num_classes = model_args['num_classes']
    running_metrics = runningScore(num_classes)
    score_dice = []
    score_dice_best = 0
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
            outputs,body_out, edge_out = model(img)            
            edge_out = edge_out[0, :, :, :]           
            loss = loss_ce(outputs, label) + loss_dc(outputs=outputs, gt=label) + loss_ce(body_out, label)+ loss_dc(outputs=body_out, gt=label)+loss_edge(edge_out, edge)#+loss_de(outputs=edge_out, gt=edge.long()) 
            
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
                outputs_all, _, _= model(img) 
                outputs = outputs_all[0, :, :, :]
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
    path = os.path.join(cur_path, 'IBSR_18')
    args = dict(PATH=path, gpu_id=1)

    args['label_source'] = (9, 10, 11, 12, 13, 17, 18, 48, 49, 50, 51, 52, 53, 54)
    args['label_target'] = (1,1,2,3,4,5,6,7,7,8,9,10,11,12)
    args['n_epoch'] = 15
    args['model_args'] = {'in_channels': 1, 'num_classes': 13}
    args['experiment_i'] =0
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

    IDs = [(['09', '15', '17', '04', '07', '08', '18', '11', '13'], ['01', '02', '03', '05', '06', '10', '12', '14', '16']),
          (['12', '01', '09', '03', '13', '02', '15', '08', '16'], ['04', '05', '06', '07', '10', '11', '14', '17', '18']),
          (['17', '04', '16', '15', '06', '01', '02', '08', '12'], ['03', '05', '07', '09', '10', '11', '13', '14', '18']),
          (['10', '06', '03', '09', '02', '18', '13', '05', '17'], ['01', '04', '07', '08', '11', '12', '14', '15', '16']),
          (['15', '08', '13', '16', '02', '03', '10', '06', '05'], ['01', '04', '07', '09', '11', '12', '14', '17', '18']),
          (['06', '08', '03', '01', '10', '16', '02', '12', '04'], ['05', '07', '09', '11', '13', '14', '15', '17', '18']),
          (['18', '03', '12', '04', '06', '05', '02', '15', '07'], ['01', '08', '09', '10', '11', '13', '14', '16', '17']),
          (['15', '14', '18', '07', '06', '16', '04', '08', '12'], ['01', '02', '03', '05', '09', '10', '11', '13', '17']),
          (['11', '10', '18', '13', '17', '14', '06', '01', '03'], ['02', '04', '05', '07', '08', '09', '12', '15', '16']),
          (['09', '18', '15', '01', '16', '13', '03', '04', '07'], ['02', '05', '06', '08', '10', '11', '12', '14', '17'])]
 
    args['IDs_train'], args['IDs_val'] = IDs[args['experiment_i']]
    train(args)
