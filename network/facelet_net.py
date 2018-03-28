import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from util import util
from util import opt
import functools
from . import base_network
from global_vars import *


class Facelet(base_network.BaseModel, nn.Module):
    def __init__(self, opt=opt.opt()):
        super(Facelet, self).__init__()
        self._default_opt()
        self.opt = self.opt.merge_opt(opt)
        self._define_model()
        self._define_optimizer()
        if self.opt.pretrained:
            state_dict = util.load_from_url(model_urls[self.opt.effect], save_dir=facelet_path)
            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)

    def _default_opt(self):
        self.opt = opt.opt()
        self.opt.save_dir = 'checkpoints'
        self.opt.name = 'Facelet'
        self.opt.lr = 1e-4
        self.opt.pretrained = True
        self.opt.effect = 'facehair'

    def _define_model(self):
        self.model = nn.Module()
        self.model.w_1 = simpleCNNGenerator(256, 256)
        self.model.w_2 = simpleCNNGenerator(512, 512)
        self.model.w_3 = simpleCNNGenerator(512, 512)

    def _define_optimizer(self):
        self.schedulers = []
        self.optimizer = optim.Adam([{'params': self.model.parameters(), 'lr': self.opt.lr}])

    def forward(self, vgg_feat):
        f1, f2, f3 = vgg_feat
        w1 = self.model.w_1(f1)
        w2 = self.model.w_2(f2)
        w3 = self.model.w_3(f3)
        return w1, w2, w3

    def backward_G(self, vgg_feat, gt):
        self.mseloss = []

        # it sets the loss
        for i in range(3):
            criterionL2 = torch.nn.MSELoss(size_average=True)
            self.mseloss += [criterionL2(vgg_feat[i], gt[i])]
        self.loss_all = sum(self.mseloss)
        self.loss_all.backward()

    def optimize_parameters(self, vgg_feat, gt):
        self.model.train()
        self.optimizer.zero_grad()
        vgg_feat = [util.toVariable(vgg_feat_, requires_grad=False).cuda() for vgg_feat_ in vgg_feat]
        gt = [util.toVariable(gt_, requires_grad=False).cuda() for gt_ in gt]
        w = self.forward(vgg_feat)
        self.backward_G(w, gt)
        self.optimizer.step()
        return w

    def get_current_errors(self):
        return OrderedDict([('loss1', self.mseloss[0].data[0]),
                            ('loss2', self.mseloss[1].data[0]),
                            ('loss3', self.mseloss[2].data[0]),
                            ('loss_all', self.loss_all.data[0]),
                            ])

    def print_current_errors(self, epoch, i):
        errors = self.get_current_errors()
        message = '(epoch: %d, iters: %d) ' % (epoch, i)
        for k, v in errors.items():
            message += '%s: %.3f ' % (k, v)
        print(message)

    def save(self, label, save_dir='checkpoints'):
        self.save_network(self.model, save_dir, label)

    def load(self, label, pretrain_path='checkpoints'):
        print('loading facelet model')
        self.load_network(self.model, pretrain_path, label)
        print('facelet model loaded successfully')


class simpleCNNGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=None, n_blocks=3):
        assert (n_blocks >= 0)
        super(simpleCNNGenerator, self).__init__()
        ngf = input_nc
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if norm_layer is not None:
            model = [nn.ReflectionPad2d(1),
                     nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0,
                               bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(inplace=True)]
        else:
            model = [nn.ReflectionPad2d(1),
                     nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0),
                     nn.ReLU(inplace=True)]
        for i in range(n_blocks - 2):
            if norm_layer is not None:
                model += [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1,
                                    bias=use_bias),
                          norm_layer(ngf),
                          nn.ReLU(inplace=True)]
            else:
                model += [nn.Conv2d(ngf, ngf, kernel_size=3, padding=1),
                          nn.ReLU(inplace=True)]

        if norm_layer is not None:
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0,
                                bias=use_bias)]
        else:
            model += [nn.ReflectionPad2d(1),
                      nn.Conv2d(ngf, output_nc, kernel_size=3, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)
