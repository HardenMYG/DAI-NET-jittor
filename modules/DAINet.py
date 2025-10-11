# -*- coding:utf-8 -*-

import jittor as jt
from jittor import nn
import numpy as np

from layers import * 
from data.config import cfg  

class Interpolate(nn.Module):
    def __init__(self, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor

    def execute(self, x):
        return nn.interpolate(x, scale_factor=self.scale_factor, mode='nearest')

class FEM(nn.Module):
    def __init__(self, in_planes):
        super().__init__()
        inter_planes = in_planes // 3
        inter_planes1 = in_planes - 2 * inter_planes
        self.branch1 = nn.Conv(in_planes, inter_planes, 3, 1, 3, dilation=3)
        self.branch2 = nn.Sequential(
            nn.Conv(in_planes, inter_planes, 3, 1, 3, dilation=3),
            nn.ReLU(),
            nn.Conv(inter_planes, inter_planes, 3, 1, 3, dilation=3)
        )
        self.branch3 = nn.Sequential(
            nn.Conv(in_planes, inter_planes1, 3, 1, 3, dilation=3),
            nn.ReLU(),
            nn.Conv(inter_planes1, inter_planes1, 3, 1, 3, dilation=3),
            nn.ReLU(),
            nn.Conv(inter_planes1, inter_planes1, 3, 1, 3, dilation=3)
        )

    def execute(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = jt.concat([x1, x2, x3], dim=1)
        out = nn.relu(out)
        return out

class DSFD(nn.Module):
    def __init__(self, phase, base, extras, fem, head1, head2, num_classes):
        super().__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.vgg = nn.ModuleList(base)
        self.L2Normof1 = L2Norm(256, 10)
        self.L2Normof2 = L2Norm(512, 8)
        self.L2Normof3 = L2Norm(512, 5)
        self.extras = nn.ModuleList(extras)
        self.fpn_topdown = nn.ModuleList(fem[0])
        self.fpn_latlayer = nn.ModuleList(fem[1])
        self.fpn_fem = nn.ModuleList(fem[2])
        self.L2Normef1 = L2Norm(256, 10)
        self.L2Normef2 = L2Norm(512, 8)
        self.L2Normef3 = L2Norm(512, 5)
        self.loc_pal1 = nn.ModuleList(head1[0])
        self.conf_pal1 = nn.ModuleList(head1[1])
        self.loc_pal2 = nn.ModuleList(head2[0])
        self.conf_pal2 = nn.ModuleList(head2[1])
        self.ref = nn.Sequential(
            nn.Conv(64, 64, 3, 1, 1),
            nn.ReLU(),
            Interpolate(2),
            nn.Conv(64, 3, 3, 1, 1),
            nn.Sigmoid()
        )
        self.KL = DistillKL(T=4.0)
        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg)

    def _upsample_prod(self, x, y):
        _, _, H, W = y.shape
        return nn.interpolate(x, size=(H, W), mode='bilinear') * y


    def test_forward(self, x, return_features=False):
        size = x.shape[2:]
        pal1_sources = []
        pal2_sources = []
        loc_pal1 = []
        conf_pal1 = []
        loc_pal2 = []
        conf_pal2 = []

        for k in range(16):
            x = self.vgg[k](x)
            if k == 4:
                x_ = x
        R = self.ref(x_[0:1])

        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)

        for k in range(2):
            x = nn.relu(self.extras[k](x))
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = nn.relu(self.extras[k](x))
        of6 = x
        pal1_sources.append(of6)

        conv7 = nn.relu(self.fpn_topdown[0](of6))
        x = nn.relu(self.fpn_topdown[1](conv7))
        conv6 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[0](of5)))
        x = nn.relu(self.fpn_topdown[2](conv6))
        convfc7_2 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[1](of4)))
        x = nn.relu(self.fpn_topdown[3](convfc7_2))
        conv5 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[2](of3)))
        x = nn.relu(self.fpn_topdown[4](conv5))
        conv4 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[3](of2)))
        x = nn.relu(self.fpn_topdown[5](conv4))
        conv3 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[4](of1)))

        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).transpose(0,2,3,1))
            conf_pal1.append(c(x).transpose(0,2,3,1))

        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).transpose(0,2,3,1))
            conf_pal2.append(c(x).transpose(0,2,3,1))

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].shape[1], loc_pal1[i].shape[2]]
            features_maps += [feat]

        loc_pal1 = jt.concat([o.reshape(o.shape[0], -1) for o in loc_pal1], dim=1)
        conf_pal1 = jt.concat([o.reshape(o.shape[0], -1) for o in conf_pal1], dim=1)
        loc_pal2 = jt.concat([o.reshape(o.shape[0], -1) for o in loc_pal2], dim=1)
        conf_pal2 = jt.concat([o.reshape(o.shape[0], -1) for o in conf_pal2], dim=1)

        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        self.priors_pal1 = priorbox.execute()

        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        self.priors_pal2 = priorbox.execute()

        if return_features:
            # 直接返回 backbone 特征（论文中的 backbone features）
            return (of1, of2, of3, of4, of5, of6)

        if self.phase == 'test':
            output = self.detect.execute(
                loc_pal2.reshape(loc_pal2.shape[0], -1, 4),
                self.softmax(conf_pal2.reshape(conf_pal2.shape[0], -1, self.num_classes)),
                self.priors_pal2
            )
        else:
            output = (
                loc_pal1.reshape(loc_pal1.shape[0], -1, 4),
                conf_pal1.reshape(conf_pal1.shape[0], -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.reshape(loc_pal2.shape[0], -1, 4),
                conf_pal2.reshape(conf_pal2.shape[0], -1, self.num_classes),
                self.priors_pal2)
        return output, R

    def execute(self, x, x_light, I, I_light):
        size = x.shape[2:]
        pal1_sources = []
        pal2_sources = []
        loc_pal1 = []
        conf_pal1 = []
        loc_pal2 = []
        conf_pal2 = []

        for k in range(5):
            x_light = self.vgg[k](x_light)

        for k in range(16):
            x = self.vgg[k](x)
            if k == 4:
                x_dark = x

        R_dark = self.ref(x_dark)
        R_light = self.ref(x_light)

        x_dark_2 = (I * R_light).stop_grad()
        x_light_2 = (I_light * R_dark).stop_grad()

        for k in range(5):
            x_light_2 = self.vgg[k](x_light_2)
        for k in range(5):
            x_dark_2 = self.vgg[k](x_dark_2)

        R_dark_2 = self.ref(x_light_2)
        R_light_2 = self.ref(x_dark_2)

        x_light_flat = x_light.reshape(x_light.shape[0], x_light.shape[1], -1).mean(dim=-1)
        x_dark_flat = x_dark.reshape(x_dark.shape[0], x_dark.shape[1], -1).mean(dim=-1)
        x_light_2_flat = x_light_2.reshape(x_light_2.shape[0], x_light_2.shape[1], -1).mean(dim=-1)
        x_dark_2_flat = x_dark_2.reshape(x_dark_2.shape[0], x_dark_2.shape[1], -1).mean(dim=-1)



        loss_mutual = cfg.WEIGHT.MC * (self.KL.execute(x_light_flat, x_dark_flat) + self.KL.execute(x_dark_flat, x_light_flat) +
                                       self.KL.execute(x_light_2_flat, x_dark_2_flat) + self.KL.execute(x_dark_2_flat, x_light_2_flat))

        of1 = x
        s = self.L2Normof1(of1)
        pal1_sources.append(s)
        for k in range(16, 23):
            x = self.vgg[k](x)
        of2 = x
        s = self.L2Normof2(of2)
        pal1_sources.append(s)

        for k in range(23, 30):
            x = self.vgg[k](x)
        of3 = x
        s = self.L2Normof3(of3)
        pal1_sources.append(s)

        for k in range(30, len(self.vgg)):
            x = self.vgg[k](x)
        of4 = x
        pal1_sources.append(of4)

        for k in range(2):
            x = nn.relu(self.extras[k](x))
        of5 = x
        pal1_sources.append(of5)
        for k in range(2, 4):
            x = nn.relu(self.extras[k](x))
        of6 = x
        pal1_sources.append(of6)

        conv7 = nn.relu(self.fpn_topdown[0](of6))
        x = nn.relu(self.fpn_topdown[1](conv7))
        conv6 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[0](of5)))
        x = nn.relu(self.fpn_topdown[2](conv6))
        convfc7_2 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[1](of4)))
        x = nn.relu(self.fpn_topdown[3](convfc7_2))
        conv5 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[2](of3)))
        x = nn.relu(self.fpn_topdown[4](conv5))
        conv4 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[3](of2)))
        x = nn.relu(self.fpn_topdown[5](conv4))
        conv3 = nn.relu(self._upsample_prod(x, self.fpn_latlayer[4](of1)))

        ef1 = self.fpn_fem[0](conv3)
        ef1 = self.L2Normef1(ef1)
        ef2 = self.fpn_fem[1](conv4)
        ef2 = self.L2Normef2(ef2)
        ef3 = self.fpn_fem[2](conv5)
        ef3 = self.L2Normef3(ef3)
        ef4 = self.fpn_fem[3](convfc7_2)
        ef5 = self.fpn_fem[4](conv6)
        ef6 = self.fpn_fem[5](conv7)

        pal2_sources = (ef1, ef2, ef3, ef4, ef5, ef6)
        for (x, l, c) in zip(pal1_sources, self.loc_pal1, self.conf_pal1):
            loc_pal1.append(l(x).transpose(0,2,3,1))
            conf_pal1.append(c(x).transpose(0,2,3,1))

        for (x, l, c) in zip(pal2_sources, self.loc_pal2, self.conf_pal2):
            loc_pal2.append(l(x).transpose(0,2,3,1))
            conf_pal2.append(c(x).transpose(0,2,3,1))

        features_maps = []
        for i in range(len(loc_pal1)):
            feat = []
            feat += [loc_pal1[i].shape[1], loc_pal1[i].shape[2]]
            features_maps += [feat]

        loc_pal1 = jt.concat([o.reshape(o.shape[0], -1) for o in loc_pal1], dim=1)
        conf_pal1 = jt.concat([o.reshape(o.shape[0], -1) for o in conf_pal1], dim=1)
        loc_pal2 = jt.concat([o.reshape(o.shape[0], -1) for o in loc_pal2], dim=1)
        conf_pal2 = jt.concat([o.reshape(o.shape[0], -1) for o in conf_pal2], dim=1)

        priorbox = PriorBox(size, features_maps, cfg, pal=1)
        self.priors_pal1 = priorbox.execute()

        priorbox = PriorBox(size, features_maps, cfg, pal=2)
        self.priors_pal2 = priorbox.execute()

        if self.phase == 'test':
            output = self.detect.execute(
                loc_pal2.reshape(loc_pal2.shape[0], -1, 4),
                self.softmax(conf_pal2.reshape(conf_pal2.shape[0], -1, self.num_classes)),
                self.priors_pal2
            )
        else:
            output = (
                loc_pal1.reshape(loc_pal1.shape[0], -1, 4),
                conf_pal1.reshape(conf_pal1.shape[0], -1, self.num_classes),
                self.priors_pal1,
                loc_pal2.reshape(loc_pal2.shape[0], -1, 4),
                conf_pal2.reshape(conf_pal2.shape[0], -1, self.num_classes),
                self.priors_pal2)
        return output, [R_dark, R_light, R_dark_2, R_light_2], loss_mutual

    def load_weights(self, base_file):
        print('Loading weights into state dict...')
        mdata = jt.load(base_file)
        weights = mdata['weight']
        epoch = mdata['epoch']
        self.load_state_dict(weights)
        print('Finished!')
        return epoch
    

    def weights_init(self, m):
        if isinstance(m, nn.Conv):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.zero_()
        if isinstance(m, nn.BatchNorm):
            m.weight.data[...] = 1
            m.bias.data.zero_()


vgg_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
           512, 512, 512, 'M']
extras_cfg = [256, 'S', 512, 128, 'S', 256]
fem_cfg = [256, 512, 512, 1024, 512, 256]

def fem_module(cfg):
    topdown_layers = []
    lat_layers = []
    fem_layers = []
    topdown_layers += [nn.Conv(cfg[-1], cfg[-1], 1, 1, 0)]
    for k, v in enumerate(cfg):
        fem_layers += [FEM(v)]
        cur_channel = cfg[len(cfg) - 1 - k]
        if len(cfg) - 1 - k > 0:
            last_channel = cfg[len(cfg) - 2 - k]
            topdown_layers += [nn.Conv(cur_channel, last_channel, 1, 1, 0)]
            lat_layers += [nn.Conv(last_channel, last_channel, 1, 1, 0)]
    return (topdown_layers, lat_layers, fem_layers)

def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(2, 2, ceil_mode=True)]
        else:
            conv2d = nn.Conv(in_channels, v, 3, 1, 1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    conv6 = nn.Conv(512, 1024, 3, 1, 3, dilation=3)
    conv7 = nn.Conv(1024, 1024, 1, 1, 0)
    layers += [conv6, nn.ReLU(), conv7, nn.ReLU()]
    return layers

def add_extras(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv(in_channels, cfg[k + 1], (1, 3)[flag], 2, 1)]
            else:
                layers += [nn.Conv(in_channels, v, (1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

def multibox(vgg, extra_layers, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [14, 21, 28, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv(vgg[v].out_channels, 4, 3, 1, 1)]
        conf_layers += [nn.Conv(vgg[v].out_channels, num_classes, 3, 1, 1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv(v.out_channels, 4, 3, 1, 1)]
        conf_layers += [nn.Conv(v.out_channels, num_classes, 3, 1, 1)]
    return (loc_layers, conf_layers)

def build_net_dark(phase, num_classes=2):
    base = vgg(vgg_cfg, 3)
    extras = add_extras(extras_cfg, 1024)
    head1 = multibox(base, extras, num_classes)
    head2 = multibox(base, extras, num_classes)
    fem = fem_module(fem_cfg)
    return DSFD(phase, base, extras, fem, head1, head2, num_classes)

class DistillKL(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

    def execute(self, y_s, y_t):
        p_s = nn.log_softmax(y_s / self.T, dim=1)
        p_t = nn.softmax(y_t / self.T, dim=1)
        kl_loss = nn.KLDivLoss(reduction='sum', log_target=False)
        loss = kl_loss(p_s, p_t) * (self.T ** 2) / y_s.shape[0]
        return loss