"""
Description:
General image functions
"""
import numpy as np
from scipy.misc import imresize
from scipy.ndimage import shift
from scipy.misc import imsave

import random
from scipy.misc import imread,imsave
import os


def image_prepare(img,size,interpolation = 'cubic'):
    """
    Select central crop, resize and convert gray to 3 channel image

    :param img: image
    :param size: image output size

    """


    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]

    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]

    img = imresize(img, size=[size, size], interp=interpolation)
    if (np.ndim(img) == 3):
        out_img = img
    else:
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img

    return out_img/255.0


def rand_shift(img,max_shift = 0 ):
    x_shift, y_shift = np.random.randint(-max_shift, max_shift + 1, size=2)
    img_shifted = shift(img, [x_shift, y_shift, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted

def const_shift(img,x_shift = 0, shift_y = 0 ):
    img_shifted = shift(img, [x_shift, shift_y, 0], prefilter=False, order=0, mode='nearest')
    return img_shifted


def image_collage(img_arrays, rows =10, border =5,save_file = None):
    img_len =img_arrays[0].shape[2]
    array_len  = img_arrays[0].shape[0]
    num_arrays =len(img_arrays)

    cols = int(np.ceil(array_len/rows))
    img_collage = np.ones([rows * (img_len + border) + border,num_arrays *cols * (img_len + border) + border, 3])

    for ind in range(array_len):
        x = (ind % cols) * num_arrays
        y = int(ind / cols)

        img_collage[border * (y + 1) + y * img_len:border * (y + 1) + (y + 1) * img_len, cols * (x + 1) + x * img_len:cols * (x + 1) +(x + num_arrays) * img_len]\
            = np.concatenate([img_arrays[i][ind] for i in range(num_arrays) ],axis=1)

    if(save_file is not None):
        imsave(save_file,img_collage)

    return img_collage



def get_ext_from_dir(ext_dir = 'data/ImageNet_Files/val' , num_samples = 100,img_len  =112):
    img_file = random.sample(os.listdir(ext_dir), num_samples)
    images = np.zeros([num_samples, img_len,img_len, 3])
    count = 0

    for file in img_file:
        img = imread(ext_dir + '/' + file)
        images[count]   = image_prepare(img, img_len)
        count += 1
    return images
