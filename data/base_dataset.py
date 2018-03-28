import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import math
import torch

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


class FitToQuantum():
    def __init__(self, quantum=112):
        self.quantum = float(quantum)

    def __call__(self, img):
        quantum = self.quantum
        size = img.size()

        if img.size(1) % int(quantum) == 0:
            pad_w = 0
        else:
            pad_w = int((quantum - img.size(1) % int(quantum)) / 2)

        if img.size(2) % int(quantum) == 0:
            pad_h = 0
        else:
            pad_h = int((quantum - img.size(2) % int(quantum)) / 2)

        res = torch.zeros(size[0],
                          int(math.ceil(size[1] / quantum) * quantum),
                          int(math.ceil(size[2] / quantum) * quantum))
        res[:, pad_w:(pad_w + size[1]), pad_h:(pad_h + size[2])].copy_(img)
        return res


def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
