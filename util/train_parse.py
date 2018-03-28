from argparse import *
from . import util
import os


class ArgumentParser(ArgumentParser):
    def __init__(self, formatter_class=ArgumentDefaultsHelpFormatter, **kwargs):
        super(ArgumentParser, self).__init__(formatter_class=formatter_class, **kwargs)
        self.add_argument('-gpu', '--gpu_id', default='7', help='gpu id')
        self.add_argument('-lr', '--lr', type=float, default=1e-4, help='learning rate')
        self.add_argument('-ss', '--snapshot', type=int, default=10000, help='number of steps to save each snapshot')
        self.add_argument('-sp', '--save_path', default='',
                          help='the save path. it will automatically stored in checkpoints/SCRIPT_NAME/SAVEPATH')
        self.add_argument('-bs', '--batch_size', type=int, default=16, help='batch_size')
        self.add_argument('--epoch', type=int, default=30, help='the number of epoches')
        self.add_argument('-ct', '--continue_train', action='store_true', help='whether continue training')

    def parse_args(self, args=None, namespace=None):
        args = super(ArgumentParser, self).parse_args(args=args, namespace=namespace)
        util.print_args_to_screen(args)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        return args
