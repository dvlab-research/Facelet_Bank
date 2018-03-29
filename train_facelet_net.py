import os
from data.trainDataset import Dataset
from data import walk_data
from torch.utils.data import DataLoader
from network.facelet_net import Facelet
import util.train_parse as argparse
from network.base_network import VGG
from util import util
import torch
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--effect', default='facehair',
                    help='What kind of effect to train. Please note that the spefify')
parser.add_argument('--pretrain_path', help='pretrain model path')
parser.add_argument('--pretrain_label', default='latest', help='pretrain model label')
parser.add_argument('--npz_path', type=str,
                    default='../face_edit/datasets/training/proj23_ycchen2/deepfeatinterp/attribute_vector',
                    help='the path of attribute vector')
parser.add_argument('-ip', '--input_path', type=str,
                    default='../face_edit/datasets/training/proj23/houyang/images_aligned/facemodel_four/celeba',
                    help='the path of training image')
args = parser.parse_args()


def train():
    image_path = args.input_path
    gt_path = args.npz_path
    # gt_path = '../face_edit/datasets/training/proj23/features/facefeature/facemodel_four/celeba'
    # image_path = '../face_edit/datasets/training/proj23/houyang/facemodel_four/celeba'
    npz_list, image_list = walk_data.glob_image_from_npz(gt_path, image_path, '*_%s.npz' % args.effect)
    trainingSet = Dataset(image_list=image_list, npz_list=npz_list)
    dataloader = DataLoader(trainingSet, batch_size=args.batch_size, shuffle=True, num_workers=4)
    vgg = torch.nn.DataParallel(VGG()).cuda()
    args.pretrained = False
    facelet = Facelet(args)
    facelet = facelet.cuda()
    global_step = 0
    if args.pretrain_path is not None:
        facelet.load(args.pretrain_path, args.pretrain_label)
    for epoch in range(args.epoch):
        for idx, data in enumerate(tqdm(dataloader), 0):
            image, gt = data
            vgg_feat = vgg.forward(util.toVariable(image).cuda())
            w = facelet.optimize_parameters(vgg_feat, gt)
            if global_step % 10 == 0:
                facelet.print_current_errors(epoch=epoch, i=idx)
            if global_step > 0 and global_step % args.snapshot == 0:
                facelet.save(label=args.effect)
            global_step += 1
        facelet.save(label=args.effect)
    facelet.save(label=args.effect)


if __name__ == '__main__':
    train()
