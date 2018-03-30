'''
This script is the testing interface. Please see readme.md for details.

If you find this project useful for your research, please kindly cite our paper:

@inproceedings{Chen2018Facelet,
  title={Facelet-Bank for Fast Portrait Manipulation},
  author={Chen, Ying-Cong and Lin, Huaijia and Shu, Michelle and Li, Ruiyu and Tao, Xin and Ye, Yangang and Shen, Xiaoyong and Jia, Jiaya},
  booktitle={CVPR},
  year={2018}
}
'''

from network.facelet_net import *
from util import test_parse as argparse
from data.testData import Dataset, VideoDataset, untransform
from torch.utils.data import DataLoader
from tqdm import tqdm
from network.decoder import vgg_decoder
from global_vars import *
import imageio
from util import framework
from network.base_network import VGG
import glob
import os


def forward(image, vgg, facelet, decoder, weight):
    vgg_feat = vgg.forward(image)
    w = facelet.forward(vgg_feat)
    vgg_feat_transformed = [vgg_feat_ + weight * w_ for vgg_feat_, w_ in zip(vgg_feat, w)]
    return decoder.forward(vgg_feat_transformed, image)


def test_image():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='test_image')
    parser.add_argument('-ip', '--input_path', default='examples/input.png',
                        help='path of the testing image. Use comma to separate each path. If this argument is a directory, then it will test all images in this directory.')
    args = parser.parse_args()
    vgg = VGG()
    args.pretrained = not args.local_model
    facelet = Facelet(args)
    if args.local_model:
        facelet.load(args.effect, pretrain_path='checkpoints')
    decoder = vgg_decoder()
    if not args.cpu:
        vgg = vgg.cuda()
        facelet = facelet.cuda()
        decoder = decoder.cuda()
    if os.path.isdir(args.input_path):
        print('input path is a directory. All images in this folder will be tested.')
        image_list = glob.glob(args.input_path + '/*')
    else:
        image_list = args.input_path.split(',')
    dataset = Dataset(image_list, scale=util.str2numlist(args.size))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, data in enumerate(tqdm(dataloader), 0):
        image, filename, image_shape = data
        filename = filename[0]
        image = util.toVariable(image)
        if not args.cpu:
            image = image.cuda()
        output = forward(image, vgg, facelet, decoder, args.strength)
        output = untransform(output.data[0].cpu())
        output = util.center_crop(output, (image_shape[0][0], image_shape[1][0]))
        imageio.imwrite('%s-%s-s-%d.%s' % (util.remove_format_name(filename), args.effect, args.strength, filename[-3:]),
                        output)


def test_video():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', help='test_video')
    parser.add_argument('-ip', '--input_path', default='examples/input.mp4', help='the path to a video file')
    args = parser.parse_args()
    vgg = VGG()
    args.pretrained = not args.local_model
    facelet = Facelet(args)
    if args.local_model:
        facelet.load(args.effect, pretrain_path='checkpoints')
    decoder = vgg_decoder()
    if not args.cpu:
        vgg = vgg.cuda()
        facelet = facelet.cuda()
        decoder = decoder.cuda()
    reader = imageio.get_reader(args.input_path)
    fps = reader.get_meta_data()['fps']
    savepath = '%s-%s-s-%d.%s' % (util.remove_format_name(args.input_path), args.effect, args.strength, args.input_path[-3:])
    print('saving to %s' % savepath)
    writer = imageio.get_writer(savepath, fps=fps)

    dataset = VideoDataset(reader, scale=util.str2numlist(args.size))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    for idx, data in enumerate(tqdm(dataloader), 0):
        image, image_shape = data
        image = util.toVariable(image)
        if not args.cpu:
            image = image.cuda()
        output = forward(image, vgg, facelet, decoder, args.strength)
        output = untransform(output.data[0].cpu())
        output = util.center_crop(output, (image_shape[0][0], image_shape[1][0]))
        writer.append_data(output)
    writer.close()


if __name__ == '__main__':
    Framework = framework.CommandCall()
    Framework.add(test_image)
    Framework.add(test_video)
    Framework.run()
