from __future__ import print_function
import os
import glob
import sys
import fnmatch
import shutil
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
import tempfile

try:
    from requests.utils import urlparse
    import requests.get as urlopen

    requests_available = True
except ImportError:
    requests_available = False
    if sys.version_info[0] == 2:
        from urlparse import urlparse  # noqa f811
        from urllib2 import urlopen  # noqa f811
    else:
        from urllib.request import urlopen
        from urllib.parse import urlparse


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


# implement expand_dims for pytorch tensor x
def expand_dims(x, axis):
    shape = list(x.size())
    assert len(shape) >= axis, 'expand_dims error'
    shape.insert(axis, 1)
    y = x.view(shape)
    return y


# convert a unknown object (could be variable) to tensor
def toTensor(obj):
    if type(obj) == Variable:
        y = obj.data
    elif type(obj) == np.ndarray:
        y = torch.from_numpy(obj)
    elif type(obj) == torch.FloatTensor or type(obj) == torch.cuda.FloatTensor:
        y = obj
    elif type(obj) == torch.nn.Parameter:
        y = obj.data
    else:
        assert 0, 'type: %s is not supported yet' % type(obj)
    return y


# convert a unknown object (could be variable) to tensor
def toVariable(obj, requires_grad=False):
    if type(obj) == Variable:
        y = Variable(obj.data, requires_grad=requires_grad)
    elif type(obj) == np.ndarray:
        y = torch.from_numpy(obj.astype(np.float32))
        y = Variable(y, requires_grad=requires_grad)
    elif type(obj) == torch.FloatTensor or type(obj) == torch.cuda.FloatTensor:
        y = Variable(obj, requires_grad=requires_grad)
    else:
        assert 0, 'type: %s is not supported yet' % type(obj)
    return y


def print_network(net, filepath=None):
    if filepath is None:
        num_params = 0
        for param in net.parameters():
            num_params += param.numel()
        print(net)
        print('Total number of parameters: %d' % num_params)
    else:
        num_params = 0
        with open(filepath + '/network.txt', 'w') as f:
            for param in net.parameters():
                num_params += param.numel()
            print(net, file=f)
            f.write('Total number of parameters: %d' % num_params)


# -------------------------- General ---------------------------------#


def _download_url_to_file(url, dst):
    u = urlopen(url)
    if requests_available:
        file_size = int(u.headers["Content-Length"])
        u = u.raw
    else:
        meta = u.info()
        if hasattr(meta, 'getheaders'):
            file_size = int(meta.getheaders("Content-Length")[0])
        else:
            file_size = int(meta.get_all("Content-Length")[0])

    f = tempfile.NamedTemporaryFile(delete=False)
    with tqdm(total=file_size) as pbar:
        while True:
            buffer = u.read(8192)
            if len(buffer) == 0:
                break
            f.write(buffer)
            pbar.update(len(buffer))
    f.close()
    shutil.move(f.name, dst)


def load_from_url(url, save_dir='facelet_bank'):
    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    cached_file = os.path.join(save_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        _download_url_to_file(url, cached_file)
    return torch.load(cached_file)


def center_crop(img, target_size):
    '''
    center crop on numpy data.
    :param img: H x W x C
    :param target_size: h x w
    :return: h x w x C
    '''
    diff_x = img.shape[0] - target_size[0]
    diff_y = img.shape[1] - target_size[1]
    start_x = int(diff_x // 2)
    start_y = int(diff_y // 2)
    if len(img.shape) > 2:
        img2 = img[start_x:start_x + target_size[0], start_y:start_y + target_size[1], :]
    else:
        img2 = img[start_x:start_x + target_size[0], start_y:start_y + target_size[1]]
    return img2


def remove_format_name(filename):
    filename = filename.split('.')
    filename = '.'.join(filename[:-1])
    return filename

def check_exist(file_list):
    for file in file_list:
        if not os.path.exists(file):
            print('file not exit: ' + file)
            return False
    return True


def str2numlist(str_in, type=int):
    dc = []
    dc_str = str_in.split(',')
    for d in dc_str:
        dc += [type(d)]
    return dc


def mkdir(path):
    if not os.path.exists(path):
        print('mkdir %s' % path)
        os.makedirs(path)


def globall(path, pattern):
    '''
    glob all data based on the pattern
    :param path: the root path
    :param pattern: the pattern to filter
    :return: all files that matches the pattern
    '''
    matches = []
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, pattern):
            matches.append(os.path.join(root, filename))
    return matches


def script_name():
    name = os.path.basename(sys.argv[0])
    name = name[:-3]
    return name


def print_args(ckpt_dir, args):
    '''
    print all args generated from the argparse
    :param ckpt_dir: the save dir
    :param args: the args
    :return:
    '''
    args_dict = vars(args)
    with open(ckpt_dir + '/options.txt', 'w') as f:
        for k, v in args_dict.items():
            f.write('%s: %s\n' % (k, v))


def print_args_to_screen(args):
    '''
    print all args generated from the argparse
    :param ckpt_dir: the save dir
    :param args: the args
    :return:
    '''
    args_dict = vars(args)
    for k, v in args_dict.items():
        print('%s: %s' % (k, v))
    print('\n')


