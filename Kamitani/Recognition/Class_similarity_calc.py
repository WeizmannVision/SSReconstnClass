#######################################################################################################################
#                                           Class_similarity_calc.py                                                  #
# This script takes images reconstructed from fMRI and classify them.                                                 #
# The outputs are: (i) numpy and csv files depicting the experiments, and (ii) a graph of the results.                #
#######################################################################################################################


import argparse
import os
import sys
import torch
import numpy as np
from skimage.transform import resize
from scipy.misc import imread,imsave
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = '2'

from torchvision import models
import torch
import torch.nn as nn
from Utils.pearsonr import pearsonr_corr, pearsonr_corr_vec
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#############################################################################
num_test_images = 50
layers = [3]
reconst_dirs = {}

in_dir =  str(sys.argv[1])
res_dir = str(sys.argv[2])
name = str(sys.argv[3])

target_classes = np.load('data/ImageNet_Files/target_classes.npy')

save_distances_file = res_dir + '/' + name + '_class_distance.npz'
save_acc_npz = res_dir + '/' + name + '_class_acc.npz'
save_acc_df = res_dir + '/' + name + '_class_acc.csv'

reconst_dirs['full'] = in_dir


num_images_per_class = 100

way_top1 = [50,100,500,1000]
way_top5 = [1000]

#############################################################################
def im2tensor(image, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))

class AlexNet_feat(nn.Module):

    def __init__(self):
        super().__init__()
        alex = models.alexnet(pretrained=True).to(device)
        features = []
        #for l in [2, 5, 7, 8, 11]:
        for l in [3, 5, 8, 10, 12]:
            features.append(torch.nn.Sequential(alex.features[:l]))

        # for l in [3, 6, 7]:
        #     features.append(torch.nn.Sequential(alex.classifier[:l]))

        self.features = features
        self.conv = [torch.nn.Sequential(alex.features)]

    def forward(self, x, layer=0):
        if (layer < 5):
            return self.features[layer](x)
        else:
            x = F.interpolate(x, size=224, mode='bilinear', align_corners=True)
            x = self.conv[0](x)
            x = x.view(x.size(0), 256 * 6 * 6)
            x = self.features[layer](x)
            return x


feat = AlexNet_feat().to(device)


def get_images(dir_, num_images = num_test_images , resolution = 112):
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




def measure_similarity_arr(dis_metric_self, dis_metric_ext=None, xway=5, num_exp=1000, top5 =False):
    results = []
    s1 = dis_metric_self.shape[0]

    if (dis_metric_ext is not None):
        s2 = dis_metric_ext.shape[1]
    else:
        s2 = s1

    correct_count = 0
    for im in range(s1):
        results_im = []
        for i in range(num_exp):
            t = np.array([im])  # t = np.random.randint(s1,size=1)
            list_r = np.delete(np.arange(s2), t)
            r_rand = np.random.choice(list_r, replace=False, size=xway - 1)
            # indexes = np.concatenate([t,r_rand])
            distance_self = dis_metric_self[t, t]
            if (dis_metric_ext is not None):
                distance_ext = dis_metric_ext[t, r_rand]
            else:
                distance_ext = dis_metric_self[t, r_rand]
            distances = np.concatenate([distance_self, distance_ext])

            if(top5 ):
                res = np.any(np.argsort(distances)[:5] ==0)
            else:
                res = (np.argmin(distances) == 0)
            correct_count += res
            results_im.append(res)
        results.append(results_im)
    return correct_count / np.float(num_exp * s1), results



################################## class image representatives - optinal ##############################################
from Utils.image_functions import image_prepare

# For Demo - Use specific target_classes

# if(target_classes is None):
#
#
#     ################# test image class examples  ######################
#     test_im = pd.read_csv('data/ImageNet_Files/imageID_test.csv', header=None)
#     dir = 'data/ImageNet_Files/wind'
#     test_classes = []
#     size = 112
#     class_images = np.zeros([num_test_images, num_images_per_class, size, size, 3])
#     count = 0
#     for test_file in list(test_im[1]):
#         folder = test_file.split('_')[0]
#         test_classes.append(folder)
#         print(folder)
#         all_files = os.listdir(dir + folder)
#
#         i = 0
#         j = 0
#         while i < num_images_per_class:
#             file = all_files[j]
#             if (file != test_file and file[-4:] == 'JPEG'):
#                 img = imread(dir + folder + '/' + file)
#                 img = image_prepare(img, size)
#                 class_images[count, i] = img
#                 i += 1
#             else:
#                 print(file, test_file)
#             j += 1
#
#         print(i,j)
#         count += 1
#
#     imagenet_dir = 'data/ImageNet_Files/train'
#
#     imagenet_dirs = os.listdir(imagenet_dir)
#     imagenet_dirs = list(set(imagenet_dirs) - set(test_classes))
#     imagenet_dirs.sort()
#     print(len(imagenet_dirs))
#     class_images_ext = np.zeros([len(imagenet_dirs) - 1, num_images_per_class, size, size, 3])
#
#     count = 0
#
#     for ext_folder in list(imagenet_dirs):
#         if (count % 10 == 0):
#             print(count)
#
#         all_files = os.listdir(imagenet_dir + ext_folder)
#         if ((len(all_files)) > 100): ### ensures directory holds images
#
#             for i in range(num_images_per_class):
#                 file = all_files[i]
#                 if (file[-4:] == 'JPEG'):
#                     img = imread(imagenet_dir + ext_folder + '/' + file)
#                     img = image_prepare(img, size)
#                     class_images_ext[count, i] = img
#             count += 1
#
#
#     target_classes = np.concatenate([class_images,class_images_ext],axis=0)*255.0
#     print(target_classes.shape)
#     np.save(res_dir + '/target_classes.npy', target_classes)
################################## Calc distances #####################################################################

reconst= {}

for key, dir_ in reconst_dirs.items():
    reconst[key] = get_images(dir_)#)



results = np.zeros([len(reconst_dirs.keys()), num_test_images, target_classes.shape[0]])
results_full = np.zeros([len(reconst_dirs.keys()), num_test_images, target_classes.shape[0], num_images_per_class])


# Extract embeddings of all test images, as well as mean embeddings of target classes.
for i,key in enumerate(reconst.keys()):
    rec = reconst[key]
    reconst_torch = torch.cat([im2tensor(rec[j]) for j in range(num_test_images)], 0).to(device)
    reconst_torch_embed = [feat.forward(reconst_torch, layer=l) for l in layers]
    for l in range(len(reconst_torch_embed)):
        reconst_torch_embed[l] = reconst_torch_embed[l].reshape([reconst_torch_embed[l].shape[0], reconst_torch_embed[l].shape[1]*reconst_torch_embed[l].shape[2]*reconst_torch_embed[l].shape[3]])
    reconst_torch_embed = torch.cat(reconst_torch_embed, 1)
    for c in range(target_classes.shape[0]):
        if(c%100==0):
            print(c)

        cla_im = torch.cat([im2tensor(target_classes[c, j]) for j in range(num_images_per_class)], 0).to(device)
        cla_im_embed = [feat.forward(cla_im, layer=l) for l in layers]
        for l in range(len(cla_im_embed)):
            cla_im_embed[l] = cla_im_embed[l].reshape([cla_im_embed[l].shape[0], cla_im_embed[l].shape[1]*cla_im_embed[l].shape[2]*cla_im_embed[l].shape[3]])
        cla_im_embed = torch.cat(cla_im_embed, 1)
        cla_im_embed_avg = cla_im_embed.mean(dim =0)
        for im in range(num_test_images):
            results[i, im, c] += 1 - pearsonr_corr(cla_im_embed_avg,reconst_torch_embed[im])
            if key == 'full':
                for j in range(num_images_per_class):
                    results_full[i, im, c,j] += 1- pearsonr_corr(cla_im_embed[j] ,reconst_torch_embed[im])



if(save_distances_file is not None):
    np.savez(save_distances_file,results = results, results_full = results_full, keys = list(reconst.keys()))

####################################### Calc classification acc ############################################################


res_dict = {}
df = pd.DataFrame(columns=['method','out_of','acc','img_ind'])


def calc_rec(dis, rec):
    for way in way_top1:
        acc_ext, res = measure_similarity_arr(dis,dis,xway=way)
        print(way, acc_ext)

        res = np.array(res)
        res_dict[rec + '_' + str(way)] = res

        for i in range(num_test_images):
            df.loc[len(df)] = [rec, str(way), res[i].mean(), i]

    for way in way_top5:
        acc_ext, res = measure_similarity_arr(dis,dis, xway=way,top5=True)
        print(way, acc_ext)

        res = np.array(res)
        res_dict[rec + '_' + str(way)+'_top5'] = res

        for i in range(num_test_images):
            df.loc[len(df)] = [rec, str(way)+'_top5', res[i].mean(), i]




# According to the extracted embeddings, find nearest neighbor and find accuracies.
for i, key in enumerate( reconst.keys()): #
    print(key)
    calc_rec(results[i], key)


df.to_csv(save_acc_df)
np.savez(save_acc_npz, **res_dict)



########################################### plot #######################################################################

from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
rcParams['figure.figsize'] = 12,8

ax = sns.barplot(x="out_of", y="acc", data=df, hue='method', ci="sd")
fig = ax.get_figure()
plt.show()