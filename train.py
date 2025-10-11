# -*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import random
import time
import jittor as jt
jt.flags.use_cuda = 1 
import argparse
from jittor import optim
from jittor.dataset import DataLoader
import numpy as np
from jittor.dataset import Dataset
from jittor.misc import make_grid
import jittor.nn as nn

from data.config import cfg
from layers.modules import MultiBoxLoss, EnhanceLoss
from data.widerface import WIDERDetection
from models.factory import build_net, basenet_factory
from models.enhancer import RetinexNet
from utils.DarkISP import Low_Illumination_Degrading
from PIL import Image

import torch
import csv


def load_vgg_pth_to_jt(jt_vgg_layers, pth_file):
    state_dict = torch.load(pth_file)

    conv_layers = [layer for layer in jt_vgg_layers if isinstance(layer, nn.Conv)]
    jt_keys = [f'{i}' for i in range(len(conv_layers))]  # 如果 pytorch 是 0,2,5,7...
    
    for idx, layer in enumerate(conv_layers):
        # PyTorch 的 key 对应顺序
        # 注意有跳号，0,2,5,7,... 这里我们做映射
        pt_weight_key = list(state_dict.keys())[idx*2]     # weight
        pt_bias_key   = list(state_dict.keys())[idx*2+1]   # bias

        pt_w = state_dict[pt_weight_key].cpu().numpy()
        pt_b = state_dict[pt_bias_key].cpu().numpy()

        # 检查 shape 是否一致
        if tuple(layer.weight.shape) != pt_w.shape:
            print(f"Skip layer {idx}: shape mismatch {layer.weight.shape} vs {pt_w.shape}")
            continue

        layer.weight[...] = jt.array(pt_w)
        layer.bias[...]   = jt.array(pt_b)

    print("Finished loading VGG weights by name mapping")


#python train.py --resume weights/dark/dsfd_checkpoint.pth

parser = argparse.ArgumentParser(
    description='DSFD face Detector Training With Jittor')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--batch_size',
                    default=1, type=int,
                    help='Batch size for training')
parser.add_argument('--model',
                    default='dark', type=str,
                    choices=['dark', 'vgg', 'resnet50', 'resnet101', 'resnet152'],
                    help='model for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=0, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-4, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')

args = parser.parse_args()


save_folder = os.path.join(args.save_folder, args.model)
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

train_dataset = WIDERDetection(cfg.FACE.TRAIN_FILE, mode='train', batch_size=1, 
        shuffle=True, drop_last=True,sample_ratio=1.0)

val_dataset = WIDERDetection(cfg.FACE.VAL_FILE, mode='val',batch_size=1, 
        shuffle=False, drop_last=True, sample_ratio=1.0)


train_loader = DataLoader(train_dataset)

val_loader = DataLoader(val_dataset)


min_loss = np.inf

def ssim(img1, img2, window_size=11, size_average=True):
    from jittor.nn import conv2d
    import math
    
    def create_window(window_size, channel):
        _1D_window = jt.array([math.exp(-(x - window_size//2)**2/float(2*1.5**2)) for x in range(window_size)])
        _2D_window = _1D_window.unsqueeze(1) * _1D_window.unsqueeze(0)
        _2D_window = _2D_window / jt.sum(_2D_window)
        _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
        return _2D_window.expand([channel, 1, window_size, window_size])
    
    (_, channel, height, width) = img1.shape
    window = create_window(window_size, channel)
    
    mu1 = conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = conv2d(img2, window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    
    C1 = 0.01**2
    C2 = 0.03**2
    eps = 1e-8
    
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2 + eps))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def train():
    per_epoch_size = len(train_dataset) // args.batch_size
    start_epoch = 0
    iteration = 0
    step_index = 0

    csv_file = os.path.join(save_folder, 'training_log.csv')

    basenet = basenet_factory(args.model)
    dsfd_net = build_net('train', cfg.NUM_CLASSES, args.model)
    net = dsfd_net
    net_enh = RetinexNet()
    net_enh.load_state_dict(jt.load(args.save_folder + 'decomp.pth'))

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        start_epoch = net.load_weights(args.resume)+1
        iteration = start_epoch * per_epoch_size
    else:
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'iteration', 'total_loss'
            ])
        pth_file = args.save_folder + basenet
        load_vgg_pth_to_jt(net.vgg, pth_file) 



    if not args.resume:
        print('Initializing weights...')
        net.extras.apply(net.weights_init)
        net.fpn_topdown.apply(net.weights_init)
        net.fpn_latlayer.apply(net.weights_init)
        net.fpn_fem.apply(net.weights_init)
        net.loc_pal1.apply(net.weights_init)
        net.conf_pal1.apply(net.weights_init)
        net.loc_pal2.apply(net.weights_init)
        net.conf_pal2.apply(net.weights_init)
        net.ref.apply(net.weights_init)

    # Scaling the lr
    lr = args.lr * np.round(np.sqrt(args.batch_size / 4), 4)
    param_group = []
    param_group += [{'params': dsfd_net.vgg.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.extras.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_topdown.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_latlayer.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.fpn_fem.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal1.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.loc_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.conf_pal2.parameters(), 'lr': lr}]
    param_group += [{'params': dsfd_net.ref.parameters(), 'lr': lr / 10.}]

    optimizer = optim.SGD(param_group, lr=lr, momentum=args.momentum,
                          weight_decay=args.weight_decay)


    criterion = MultiBoxLoss(cfg)
    criterion_enhance = EnhanceLoss()
    print('Loading wider dataset...')
    # print(f"验证集标注文件路径：{cfg.FACE.VAL_FILE}")  # 查看路径是否正确
    # print(f"验证集标注文件是否存在：{os.path.exists(cfg.FACE.VAL_FILE)}")  # 确认文件存在
    # print(f"验证集采样后样本数：{len(val_dataset)}")  # 关键！若为0，说明解析失败
    # print(f"val_loader 批次数量：{len(val_loader)}") 
    print('Using the specified args:')
    print(args)

    for step in cfg.LR_STEPS:
        if iteration > step:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)
    
    net_enh.eval()
    net.train()
    
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0

        for batch_idx, (images, targets, _) in enumerate(train_loader):
            images = images / 255.
            targetss = [ann for ann in targets]
            
            img_dark = jt.zeros((images.shape[0], images.shape[1], images.shape[2], images.shape[3]))
            
            # Generation of degraded data and AET groundtruth
            for i in range(images.shape[0]):
                img_dark_i, _ = Low_Illumination_Degrading(images[i])
                img_dark[i] = img_dark_i

            if iteration in cfg.LR_STEPS:
                step_index += 1
                adjust_learning_rate(optimizer, args.gamma, step_index)

            t0 = time.time()
            with jt.no_grad():
                R_dark_gt, I_dark = net_enh(img_dark)
                R_light_gt, I_light = net_enh(images)

            out, out2, loss_mutual = net(img_dark, images, I_dark, I_light)
            R_dark, R_light, R_dark_2, R_light_2 = out2

            # backprop
            optimizer.zero_grad()

            loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targetss)
            loss_l_pa12, loss_c_pal2 = criterion(out[3:], targetss)

            loss_enhance = criterion_enhance([R_dark, R_light, R_dark_2, R_light_2, I_dark, I_light], images, img_dark) * 0.1
            loss_enhance2 = nn.l1_loss(R_dark, R_dark_gt) + nn.l1_loss(R_light, R_light_gt) + (
                        1. - ssim(R_dark, R_dark_gt)) + (1. - ssim(R_light, R_light_gt))  #ref

            loss = loss_l_pa1l + loss_c_pal1 + loss_l_pa12 + loss_c_pal2 + loss_enhance2 + loss_enhance + loss_mutual #mfa
            optimizer.backward(loss)
            optimizer.clip_grad_norm( max_norm=35, norm_type=2)
            optimizer.step()
            t1 = time.time()
            losses += loss.item()

            if iteration % 100 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> pal1 conf loss:{:.4f} || pal1 loc loss:{:.4f}'.format(
                    loss_c_pal1.item(), loss_l_pa1l.item()))
                print('->> pal2 conf loss:{:.4f} || pal2 loc loss:{:.4f}'.format(
                    loss_c_pal2.item(), loss_l_pa12.item()))
                print('->> enhance loss:{:.4f}'.format(loss_enhance.item()))
                print('->> enhance2 loss:{:.4f}'.format(loss_enhance2.item()))
                print('->> mutual loss:{:.4f}'.format(loss_mutual.item()))
                print('->>lr:{}'.format(optimizer.param_groups[0]['lr']))

                with open(csv_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch, iteration, tloss
                    ])

            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                file = 'dsfd_' + repr(iteration) + '.pth'
                jt.save(dsfd_net.state_dict(),
                           os.path.join(save_folder, file))
            iteration += 1
        
        if (epoch + 1) >= 0:
            val(epoch, net, dsfd_net, net_enh, criterion)
        if iteration >= cfg.MAX_STEPS:
            break


def val(epoch, net, dsfd_net, net_enh, criterion):
    net.eval()
    step = 0
    losses = 0.
    t1 = time.time()

    for batch_idx, (images, targets, img_paths) in enumerate(val_loader):
        images = images / 255.
            
        img_dark = jt.stack([Low_Illumination_Degrading(images[i])[0] for i in range(images.shape[0])])
        out, R = net.test_forward(img_dark)

        loss_l_pa1l, loss_c_pal1 = criterion(out[:3], targets)
        loss_l_pa12, loss_c_pal2 = criterion(out[3:], targets)
        loss = loss_l_pa12 + loss_c_pal2

        losses += loss.item()
        step += 1

    tloss = losses / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        jt.save(dsfd_net.state_dict(), os.path.join(
            save_folder, 'dsfd.pth'))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': dsfd_net.state_dict(),
    }
    jt.save(states, os.path.join(save_folder, 'dsfd_checkpoint.pth'))


def adjust_learning_rate(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * gamma


if __name__ == '__main__':
    train()