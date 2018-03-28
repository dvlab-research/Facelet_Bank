#!/usr/bin/env python2

from __future__ import division
from __future__ import with_statement
from __future__ import print_function
from collections import OrderedDict
import torch
import torch.nn as nn
import os
from global_vars import *


class BaseModel(object):
    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self, x):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def print_current_errors(self, epoch, i, record_file=None):
        errors = self.get_current_errors()
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)
        if record_file is not None:
            with open(record_file + '/loss.txt', 'w') as f:
                print(message, file=f)

    def save(self, label):
        pass

    def load(self, pretrain_path, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, save_dir, label):
        save_filename = '%s.pth' % label
        save_path = os.path.join(save_dir, save_filename)
        print('saving %s in %s' % (save_filename, save_path))
        torch.save(network.cpu().state_dict(), save_path)
        network.cuda()

    # helper resuming function that can be used by subclasses
    def resume_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.opt.save_dir, save_filename)
        print('loading %s from %s' % (save_filename, save_path))
        network.load_state_dict(torch.load(save_path))

    # helper loading function that can be used by subclasses
    def load_network(self,network, pretrain_path, label):
        filename = '%s.pth' % label
        save_path = os.path.join(pretrain_path, filename)
        print('loading %s from %s' % (filename, pretrain_path))
        network.load_state_dict(torch.load(save_path))



    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)


class VGG(nn.Module, BaseModel):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.features_1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv2d(3, 64, kernel_size=3, padding=1)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv2d(64, 64, kernel_size=3, padding=1)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2_1', nn.Conv2d(64, 128, kernel_size=3, padding=1)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv2d(128, 128, kernel_size=3, padding=1)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv3_1', nn.Conv2d(128, 256, kernel_size=3, padding=1)),
            ('relu3_1', nn.ReLU(inplace=True)),
        ]))
        self.features_2 = nn.Sequential(OrderedDict([
            ('conv3_2', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('conv3_4', nn.Conv2d(256, 256, kernel_size=3, padding=1)),
            ('relu3_5', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool2d(2, 2)),
            ('conv4_1', nn.Conv2d(256, 512, kernel_size=3, padding=1)),
            ('relu4_1', nn.ReLU(inplace=True)),
        ]))
        self.features_3 = nn.Sequential(OrderedDict([
            ('conv4_2', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('conv4_4', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('relu4_4', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool2d(2, 2)),
            ('conv5_1', nn.Conv2d(512, 512, kernel_size=3, padding=1)),
            ('relu5_1', nn.ReLU(inplace=True)),
        ]))
        if pretrained:
            print('loading pretrained weights of VGG encoder')
            state_dict = torch.utils.model_zoo.load_url(model_urls['vgg19g'], 'facelet_bank')
            model_dict = self.state_dict()
            model_dict.update(state_dict)
            self.load_state_dict(model_dict)

    def forward(self, x):
        features_1 = self.features_1(x)
        features_2 = self.features_2(features_1)
        features_3 = self.features_3(features_2)
        return features_1, features_2, features_3


class Vgg_recon(nn.Module):
    def __init__(self, drop_rate=0):
        super(Vgg_recon, self).__init__()

        self.recon5 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate)
        self.upool4 = _TransitionUp(512, 512)
        self.upsample4 = _Upsample(512, 512)
        # self.recon4 = _PoolingBlock(3, 1024, 512, drop_rate = drop_rate)
        self.recon4 = _PoolingBlock(3, 512, 512, drop_rate=drop_rate)
        self.upool3 = _TransitionUp(512, 256)
        self.upsample3 = _Upsample(512, 256)
        self.recon3 = _PoolingBlock(3, 256, 256, drop_rate=drop_rate)
        self.upool2 = _TransitionUp(256, 128)
        self.upsample2 = _Upsample(256, 128)
        self.recon2 = _PoolingBlock(2, 128, 128, drop_rate=drop_rate)
        self.upool1 = _TransitionUp(128, 64)
        self.upsample1 = _Upsample(128, 64)
        self.recon1 = _PoolingBlock(1, 64, 64, drop_rate=drop_rate)
        self.recon0 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, fy):
        # print('fy', len(fy))
        features_1, features_2, features_3 = fy

        recon5 = self.recon5(features_3)
        recon5 = nn.functional.upsample(recon5, scale_factor=2, mode='bilinear')
        upool4 = self.upsample4(recon5)

        recon4 = self.recon4(upool4 + features_2)
        recon4 = nn.functional.upsample(recon4, scale_factor=2, mode='bilinear')
        upool3 = self.upsample3(recon4)

        recon3 = self.recon3(upool3 + features_1)
        recon3 = nn.functional.upsample(recon3, scale_factor=2, mode='bilinear')
        upool2 = self.upsample2(recon3)

        recon2 = self.recon2(upool2)
        recon2 = nn.functional.upsample(recon2, scale_factor=2, mode='bilinear')
        upool1 = self.upsample1(recon2)

        recon1 = self.recon1(upool1)
        recon0 = self.recon0(recon1)
        return recon0



class _PoolingBlock(nn.Sequential):
    def __init__(self, n_convs, n_input_filters, n_output_filters, drop_rate):
        super(_PoolingBlock, self).__init__()
        for i in range(n_convs):
            self.add_module('conv.%d' % (i + 1),
                            nn.Conv2d(n_input_filters if i == 0 else n_output_filters, n_output_filters, kernel_size=3,
                                      padding=1))
            # self.add_module('norm.%d' % (i+1), nn.BatchNorm2d(n_output_filters)) # xtao
            self.add_module('norm.%d' % (i + 1), nn.BatchNorm2d(n_output_filters))
            self.add_module('relu.%d' % (i + 1), nn.ReLU(inplace=True))
            if drop_rate > 0:
                self.add_module('drop.%d' % (i + 1), nn.Dropout(p=drop_rate))


class _TransitionUp(nn.Sequential):
    def __init__(self, n_input_filters, n_output_filters):
        super(_TransitionUp, self).__init__()
        self.add_module('unpool.conv',
                        nn.ConvTranspose2d(n_input_filters, n_output_filters, kernel_size=4, stride=2, padding=1))
        # self.add_module('interp.conv', nn.Conv2d(n_input_filters, n_output_filters, kernel_size=3, padding=1))
        self.add_module('unpool.norm', nn.BatchNorm2d(n_output_filters))


class _Upsample(nn.Sequential):
    def __init__(self, n_input_filters, n_output_filters):
        super(_Upsample, self).__init__()
        # self.add_module('unpool.conv', nn.ConvTranspose2d(n_input_filters, n_output_filters, kernel_size=4, stride=2, padding=1))
        self.add_module('interp.conv', nn.Conv2d(n_input_filters, n_output_filters, kernel_size=3, padding=1))
        self.add_module('interp.norm', nn.BatchNorm2d(n_output_filters))


