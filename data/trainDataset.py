'''
load paired image and npz data
'''
from . import base_dataset
import imageio
import scipy.misc
import numpy as np
import torchvision as tv
import torchvision.transforms as transforms
import torch

mean = torch.Tensor((0.485, 0.456, 0.406))
stdv = torch.Tensor((0.229, 0.224, 0.225))

forward_transform = tv.transforms.Compose(
    [transforms.ToTensor(), tv.transforms.Normalize(mean=mean, std=stdv), base_dataset.FitToQuantum()])


def vec2featmap(cur_npz, scale):
    scale1 = int(scale / 4)
    scale2 = int(scale1 / 2)
    scale3 = int(scale2 / 2)
    npz = [torch.zeros(0) for i in range(3)]
    cur_npz = torch.from_numpy(cur_npz)
    npz[0] = torch.cat((npz[0], cur_npz[: 256 * scale1 * scale1].resize_(256, scale1, scale1)), 0)
    npz[1] = torch.cat((npz[1], cur_npz[
                                256 * scale1 * scale1: 256 * scale1 * scale1 + 512 * scale2 * scale2].resize_(
        512, scale2, scale2)), 0)
    npz[2] = torch.cat((npz[2], cur_npz[
                                256 * scale1 * scale1 + 512 * scale2 * scale2: 256 * scale1 * scale1 + 512 * scale2 * scale2 + 512 * scale3 * scale3].resize_(
        512, scale3, scale3)), 0)
    return npz


class Dataset(base_dataset.BaseDataset):
    def __init__(self, image_list, npz_list, transform=forward_transform, scale=448):
        super(Dataset, self).__init__()
        self.files = image_list
        self.files_npz = npz_list
        print('* Total Images: {}'.format(len(self.files)))
        self.transform = transform
        self.scale = scale

    def __getitem__(self, index):
        try:
            img = imageio.imread(self.files[index]).astype(np.float32)
            # print('scale=',self.scale,self.files_npz[index])
            if img.shape[1] != self.scale:
                img = scipy.misc.imresize(img, [self.scale, self.scale])
            # print('img',img.shape)
            img = self.transform(img)
            # print('img_trans',img.shape)
            cur_npz = np.load(self.files_npz[index])['WF']
            scale1 = int(self.scale / 4)
            scale2 = int(scale1 / 2)
            scale3 = int(scale2 / 2)
            npz = [torch.zeros(0) for i in range(3)]
            cur_npz = torch.from_numpy(cur_npz)

            # print('npz', npz[0].shape, npz[1].shape, npz[2].shape)
            # print('cur_npz', cur_npz.shape)
            npz[0] = torch.cat((npz[0], cur_npz[: 256 * scale1 * scale1].view(256, scale1, scale1)), 0)
            npz[1] = torch.cat((npz[1], cur_npz[
                                        256 * scale1 * scale1: 256 * scale1 * scale1 + 512 * scale2 * scale2].view(
                512, scale2, scale2)), 0)
            npz[2] = torch.cat((npz[2], cur_npz[
                                        256 * scale1 * scale1 + 512 * scale2 * scale2: 256 * scale1 * scale1 + 512 * scale2 * scale2 + 512 * scale3 * scale3].view(
                512, scale3, scale3)), 0)
            # print(npz[0].size())
            return img, npz
        except:
            print('load current data error, randomly sample another one')
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.files)


def untransform(img, mean=mean, stdv=stdv):
    scale = list(img.size())
    img *= stdv.view(3, 1, 1).expand(3, scale[-2], scale[-1])
    img += mean.view(3, 1, 1).expand(3, scale[-2], scale[-1])
    img = img.numpy()
    img[img > 1.] = 1.
    img[img < 0.] = 0.
    img = img * 255
    img = img.transpose(1, 2, 0).astype(np.uint8)
    return img


