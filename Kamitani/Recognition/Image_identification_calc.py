import argparse
import os
import sys
from IPython import embed


from scipy.misc import imresize
import random
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

from skimage.transform import resize
from scipy.misc import imread,imsave
import numpy as np
from Utils.image_functions import image_prepare, get_ext_from_dir

import Utils.PerceptualSimilarity.util.util as util
import Utils.PerceptualSimilarity.models.dist_model as dm
from Utils.recognition_func import *

NUM_EXT_IMAGES = 5000



## Initializing the model
model = dm.DistModel()
model.initialize(model='net',net='vgg',use_gpu=True)

in_dir =  str(sys.argv[1])
res_dir = str(sys.argv[2])
name = str(sys.argv[3])

def compute_dis(imgs1,imgs2):
    arr = np.zeros([imgs1.shape[0], imgs2.shape[0]])
    for i in range(imgs1.shape[0]):
        for j in range(imgs2.shape[0]):
            img0 = util.im2tensor(imgs1[i])
            img1 = util.im2tensor(imgs2[j])
            dist01 = model.forward(img0, img1)
            arr[i, j] = dist01
    return arr


def get_images(dir_, num_images = 50 , resolution = 112):
    arr = np.ones([num_images, resolution, resolution, 3])
    for i in range(num_images):
        file = dir_ + 'img_' + str(i) + '.jpg'
        img = imread(file)
        if img.shape[0] != img.shape[1]:
            img = img[:,resolution:,:]
        if(img.shape[0] != resolution):
            img = resize(img, (resolution, resolution), anti_aliasing=True)
        arr[i] = img[:, :]

    return arr


############################## load images ###################################################

dir ='data/ImageNet_Files/'

results = {}
cmp_images = {}


file = np.load(dir + 'images/images_112.npz')
X_test = file['test_images']


cmp_images['target_112'] = X_test*255.0

cmp_images['ext_112'] = get_ext_from_dir(num_samples = NUM_EXT_IMAGES,img_len  =112)*255.0

results['full'] = get_images(in_dir)

output = {}
output_ext = {}



############################## compute dis  ###################################################

for  t in results.keys():
    print(t)
    output[t] =  compute_dis(results[t], cmp_images['target_112'])
    output_ext[t] = compute_dis(results[t], cmp_images['ext_112'])

np.savez(res_dir + '/' + str(name) + '_distance_test.npz',**output)
np.savez(res_dir + '/' + str(name) + '_distance_ext.npz',**output_ext)



############################## run experiments ###################################################

df = pd.DataFrame(columns=['method','out_of','acc','img_id'])
res_dict = {}

for rec in results.keys():
    print(rec)
    for metric in ['corr']:
        dis = output[rec]
        dis_ext = output_ext[rec]
        for way in [2,5,10,50]:
            acc_ext, res = measure_similarity_arr(dis,dis_ext,xway=way)
            res = np.array(res)
            print(way, acc_ext)
            res_dict[rec+'_'+str(way)] = res
            for i in range(50):
                df.loc[len(df)] = [rec,  way, res[i].mean(),i]

df.to_csv(res_dir + '/' + str(name) + '_Recognition.csv')
np.savez(res_dir + '/' + str(name) + '_Recognition.npz',**res_dict)