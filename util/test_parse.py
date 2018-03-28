from argparse import *
from . import util
import os


class ArgumentParser(ArgumentParser):
    def __init__(self, formatter_class=ArgumentDefaultsHelpFormatter, **kwargs):
        super(ArgumentParser, self).__init__(formatter_class=formatter_class, **kwargs)
        self.add_argument('-gpu', '--gpu_id', default='0', help='GPU ID to use.')
        self.add_argument('-s', '--strength', type=float, default=5, help='The edit strength')
        self.add_argument('-e', '--effect', default='facehair',
                          help='Which kind of effect to use. Current version supports facehair, older, younger.')
        self.add_argument('--size', default='0,0',
                          help='Resize the input image specific size. 0,0 indicates keep the original shape.')
        self.add_argument('-cpu', action='store_true', help='include \"-cpu\" if you want to work on CPU.')
        self.add_argument('--local_model', action='store_true', help='include \"--local_model\" if you want to test your own model. Those models are stored in \"checkpoints\"')

    def parse_args(self, args=None, namespace=None):
        args = super(ArgumentParser, self).parse_args(args=args, namespace=namespace)
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
        return args
