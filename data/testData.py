'''
only loop the images, used for testing
'''
from . import base_dataset
import scipy.misc
import imageio
import numpy as np
import torch
from .trainDataset import forward_transform, untransform

mean = torch.Tensor((0.485, 0.456, 0.406))
stdv = torch.Tensor((0.229, 0.224, 0.225))


class Dataset(base_dataset.BaseDataset):
    def __init__(self, image_list, transform=forward_transform, scale=(448, 448)):
        super(Dataset, self).__init__()
        self.files = []
        supported_format = ['jpg', 'png', 'jpeg']
        for image_now in image_list:  # filter out files that are not image
            format = image_now.split('.')[-1]
            format = format.lower()
            is_image = False
            for sf in supported_format:
                if format == sf:
                    is_image = True
                    break
            if is_image:
                self.files += [image_now]

        print('* Total Images: {}'.format(len(self.files)))
        self.transform = transform
        self.scale = scale

    def __getitem__(self, index):
        img = imageio.imread(self.files[index]).astype(np.float32)
        if self.scale[0] > 0 and img.shape[1] != self.scale:
            img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
        shape = img.shape
        img = self.transform(img)
        return img, self.files[index], shape

    def __len__(self):
        return len(self.files)


class VideoDataset(base_dataset.BaseDataset):
    def __init__(self, reader, transform=forward_transform, scale=(448, 448)):
        super(VideoDataset, self).__init__()
        self.reader = reader
        print('* Total Frames: {}'.format(len(self.reader)))
        self.transform = transform
        self.scale = scale

    def __getitem__(self, index):
        img = self.reader.get_data(index).astype(np.float32)
        if self.scale[0] > 0 and img.shape[1] != self.scale:
            img = scipy.misc.imresize(img, [self.scale[0], self.scale[1]])
        shape = img.shape
        img = self.transform(img)
        return img, shape

    def __len__(self):
        return len(self.reader)
