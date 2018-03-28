from torch import nn
import torch
from util import util
from global_vars import *
from . import base_network


class vgg_decoder(base_network.BaseModel, nn.Module):
    def __init__(self, pretrained=True):
        super(vgg_decoder, self).__init__()
        self._define_model()
        if pretrained:
            print('loading pretrained weights of VGG decoder')
            state_dict = torch.utils.model_zoo.load_url(model_urls['vgg_decoder_res'], model_dir='facelet_bank')
            model_dict = self.model.state_dict()
            model_dict.update(state_dict)
            self.model.load_state_dict(model_dict)

    def _define_model(self):
        self.model = base_network.Vgg_recon()

    def forward(self, fy, img=None):
        fy = [util.toVariable(f) for f in fy]
        y = self.model.forward(fy)
        y = y + img
        return y

    def load(self, pretrain_path, epoch_label='latest'):
        self.load_network(pretrain_path, self.model, 'recon', epoch_label)
