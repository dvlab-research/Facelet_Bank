'''
This script aims to extract attribute vectors from images. It is modified by the demo2.py.

NOTE: it works with the DFI project (https://github.com/paulu/deepfeatinterp) rather than this project.
Please copy this script to the root folder of DFI project before runing it.

Please use
python demo2_facelet.py -h
to see the help of options.

For more details, please see readme.md.
'''
from __future__ import division
from __future__ import with_statement
from __future__ import print_function

import time

timestamp = int(round(time.time()))
import numpy
import deepmodels
import json
import os.path
import argparse
import alignface
import imageutils
import utils
from tqdm import tqdm
import glob
def fit_submanifold_landmarks_to_image(template, original, Xlm, face_d, face_p, landmarks=list(range(68))):
    '''
    Fit the submanifold to the template and take the top-K.

    Xlm is a N x 68 x 2 list of landmarks.
    '''
    lossX = numpy.empty((len(Xlm),), dtype=numpy.float64)
    MX = numpy.empty((len(Xlm), 2, 3), dtype=numpy.float64)
    nfail = 0
    for i in range(len(Xlm)):
        lm = Xlm[i]
        try:
            M, loss = alignface.fit_face_landmarks(Xlm[i], template, landmarks=landmarks, image_dims=original.shape[:2])
            lossX[i] = loss
            MX[i] = M
        except alignface.FitError:
            lossX[i] = float('inf')
            MX[i] = 0
            nfail += 1
    if nfail > 1:
        print('fit submanifold, {} errors.'.format(nfail))
    a = numpy.argsort(lossX)
    return a, lossX, MX


if __name__ == '__main__':
    # configure by command-line arguments
    parser = argparse.ArgumentParser(description='Extracting attribute vector for facelet training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--effect', type=str, default='facehair', help='desired transformation')
    parser.add_argument('-ip', '--input_path', type=str,default='images/celeba', help='the training image folder')
    parser.add_argument('-gpu', type=str, default='0', help='the gpu id to use')
    parser.add_argument('--backend', type=str, default='torch', choices=['torch', 'caffe+scipy'],
                        help='reconstruction implementation')
    parser.add_argument('--K', type=int, default=100, help='number of nearest neighbors')
    parser.add_argument('--delta', type=str, default='3.5', help='comma-separated list of interpolation steps')
    parser.add_argument('--npz_path', type=str, default='attribute_vector', help='the path to store npz data')
    config = parser.parse_args()
    # print(json.dumps(config.__dict__))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu

    # load models
    if config.backend == 'torch':
        import deepmodels_torch

        model = deepmodels_torch.vgg19g_torch(device_id=0)
    elif config.backend == 'caffe+scipy':
        model = deepmodels.vgg19g(device_id=0)
    else:
        raise ValueError('Unknown backend')
    classifier = deepmodels.facemodel_attributes()
    fields = classifier.fields()
    gender = fields.index('Male')
    smile = fields.index('Smiling')
    face_d, face_p = alignface.load_face_detector()
    # Set the free parameters
    K = config.K
    delta_params = [float(x.strip()) for x in config.delta.split(',')]
    image_list = glob.glob(config.input_path+'/*')
    X = image_list
    t0 = time.time()
    opathlist = []
    # for each test image
    for i in tqdm(range(len(X))):
        xX = X[i]
        prefix_path = os.path.splitext(xX)[0]
        try:
            template, original = alignface.detect_landmarks(xX, face_d, face_p)
        except:
            print('%s face landmark detection error'% xX)
            continue
        image_dims = original.shape[:2]
        XF = model.mean_F([original])
        XA = classifier.score([xX])[0]

        # select positive and negative sets based on gender and mouth
        # You can add other attributes here.
        if config.effect == 'older':
            cP = [(gender, XA[gender] >= 0), (fields.index('Young'), True)]
            cQ = [(gender, XA[gender] >= 0), (fields.index('Young'), False)]
        elif config.effect == 'younger':
            cP = [(gender, XA[gender] >= 0), (fields.index('Young'), False)]
            cQ = [(gender, XA[gender] >= 0), (fields.index('Young'), True)]
        elif config.effect == 'facehair':
            cP = [(gender, XA[gender] >= 0), (fields.index('No_Beard'), True), (fields.index('Mustache'), False)]
            cQ = [(gender, XA[gender] >= 0), (fields.index('No_Beard'), False), (fields.index('Mustache'), True)]
        else:
            raise ValueError('Unknown method')
        P = classifier.select(cP, XA)
        Q = classifier.select(cQ, XA)
        if len(P) < 4 * K or len(Q) < 4 * K:
            print('{}: Not enough images in database (|P|={}, |Q|={}).'.format(xX, len(P), len(Q)))
            continue

        # fit the best 4K database images to input image
        Plm = classifier.lookup_landmarks(P[:4 * K])
        Qlm = classifier.lookup_landmarks(Q[:4 * K])
        idxP, lossP, MP = fit_submanifold_landmarks_to_image(template, original, Plm, face_d, face_p)
        idxQ, lossQ, MQ = fit_submanifold_landmarks_to_image(template, original, Qlm, face_d, face_p)
        # Use the K best fitted images
        xP = [P[i] for i in idxP[:K]]
        xQ = [Q[i] for i in idxQ[:K]]
        PF = model.mean_F(utils.warped_image_feed(xP, MP[idxP[:K]], image_dims))
        QF = model.mean_F(utils.warped_image_feed(xQ, MQ[idxQ[:K]], image_dims))
        WF = (QF - PF)
        if not os.path.exists(config.npz_path):
            os.makedirs(config.npz_path)
        file_name = os.path.basename(xX)
        file_name = file_name.split('.')[:-1]
        file_name = '.'.join(file_name)
        numpy.savez('{}/{}_{}.npz'.format(config.npz_path, file_name, config.effect), WF=WF)
