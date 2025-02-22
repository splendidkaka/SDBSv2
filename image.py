import cv2
import h5py
import numpy as np
from PIL import Image
import scipy.io as io
import scipy
# def load_data(img_path, args, train=True):
#     gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
#
#     while True:
#         try:
#             gt_file = h5py.File(gt_path)
#             k = np.asarray(gt_file['kpoint'])
#             img = np.asarray(gt_file['image'])
#             img = Image.fromarray(img, mode='RGB')
#             break
#         except OSError:
#             #print("path is wrong", gt_path)
#             cv2.waitKey(1000)  # Wait a bit
#     img = img.copy()
#     k = k.copy()
#
#     return img, k


def load_data(img_path, args, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map_crop')
    img = Image.open(img_path).convert('RGB')
    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            # img = np.asarray(gt_file['image'])
            # img = Image.fromarray(img, mode='RGB')
            break
        except OSError:
            #print("path is wrong", gt_path)
            cv2.waitKey(1000)  # Wait a bit
    img = img.copy()
    k = k.copy()

    return img, k


def load_data_test(img_path, args, train=True):

    img = Image.open(img_path).convert('RGB')

    return img

def load_data_fidt(img_path, args, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'gt_detr_map')
    img = Image.open(img_path).convert('RGB')

    while True:
        try:
            gt_file = h5py.File(gt_path)
            k = np.asarray(gt_file['kpoint'])
            fidt_map = np.asarray(gt_file['image'])
            break
        except OSError:
            print("path is wrong, can not load ", img_path)
            cv2.waitKey(1000)  # Wait a bit

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map,k
