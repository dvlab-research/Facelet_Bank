from util import util
import glob
import os


def filter_not_exist(npz_path, img_path):
    '''
    given a tuple of path list, filter those non-exist items
    :param tuple_list:
    :return:
    '''
    out_npz = []
    out_img = []
    for npz, img in zip(npz_path, img_path):
        if os.path.exists(npz) and os.path.exists(img):
            out_npz += [npz]
            out_img += [img]
    return out_npz, out_img

def glob_image_from_npz(npz_path, img_path, npz_pattern):
    '''
    glob images and npz based on the npz files
    :param npz_path:
    :param img_path:
    :param npz_pattern:
    :param method:
    :return:
    '''

    def npz_name_to_image_name(npz_name):
        img_name = npz_name.split('_')[0]
        # img_name = npz_name[:-13]  # 'facehair.npz'
        return img_name

    npz_list = util.globall(npz_path, npz_pattern)
    assert util.check_exist(npz_list), 'file does not exist'
    print('length of file:%d'%len(npz_list))
    non_root_name = [name[len(npz_path):] for name in npz_list]
    img_list = []
    for name in non_root_name:
        img_now = img_path + '/' + npz_name_to_image_name(name) + '.jpg'
        if not os.path.exists(img_now):
            img_now = img_path + '/' + npz_name_to_image_name(name) + '.png'
        # if not os.path.exists(img_now):
        #     img_now = img_path + '/' + npz_name_to_image_name(name) + '.jpeg'
        # if not os.path.exists(img_now):
        #     img_now = img_path + '/' + npz_name_to_image_name(name) + '.JPG'
        # if not os.path.exists(img_now):
        #     img_now = img_path + '/' + npz_name_to_image_name(name) + '.PNG'
        # if not os.path.exists(img_now):
        #     img_now = img_path + '/' + npz_name_to_image_name(name) + '.JPEG'
        img_list += [img_now]
    npz_list, img_list = filter_not_exist(npz_list, img_list)
    assert util.check_exist(img_list), 'image not exist'
    return npz_list, img_list
