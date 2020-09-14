
import scipy.misc
import numpy as np
import skimage.color
import os
from Utils.image_loss import *

#img_len = 112


def save_model_results(model,Y,X_orig,Y_test,X_test_orig,labels,Y_test_avg,Y_test_median,folder = '',img_len=112
                       ,X_ext_full=None,X_ext_part=None,pred_func =None,pred_func_enc_dec =None):
    if not os.path.exists(folder):
        os.makedirs(folder)

    if(pred_func is not None):

        X_test_avg = pred_func(model,Y_test_avg)
        Y_test_median =  pred_func(model,Y_test_median)
        X = pred_func(model,Y[0:50])
        X_test = pred_func(model,Y_test)

    else:
        X_test_avg    = model.predict(Y_test_avg)
        Y_test_median = model.predict(Y_test_median)
        X = model.predict(Y[0:50])
        X_test = model.predict(Y_test, batch_size=32)

    if(X_ext_full is not None):
        X_part_pred =pred_func_enc_dec(model,X_ext_full[0:50])
        #print(X_part_pred.shape,X_ext_part[0:50].shape,X_ext_full[0:50].shape)
        save_image_arr(X_part_pred, X_ext_part[0:50], folder=folder + 'ext_img/')
    if(pred_func_enc_dec is not None):
        X_encdec_pred = pred_func_enc_dec(model, X_orig[0:50])
        save_image_arr(X_encdec_pred, X_orig[0:50], folder=folder + 'train_encdec_img/')
        X_test_encdec_pred = pred_func_enc_dec(model, X_test_orig[0:50])
        save_image_arr(X_test_encdec_pred, X_test_orig[0:50], folder=folder + 'test_encdec_img/')

    save_image_arr(X_test_avg,X_test_orig, folder=folder+'test_avg/')
    save_image_arr(Y_test_median,X_test_orig, folder=folder+'test_median/')


    save_image_arr(X, X_orig,folder+'train/')
    print('done')


def save_imag_arr(X,X_orig,labels,dir,img_len=112):
    arr = np.ones([50 * img_len , 11* img_len , 3])
    for i in range(50):
        arr[img_len*i:img_len*(i+1),0:img_len] = X_orig[i]
        for j in range(10):
            arr[img_len * i:img_len * (i + 1), img_len * (j+1):img_len * (j + 2)] = X[labels==i][j]

    scipy.misc.imsave(dir+'imag.jpg', arr)




def image_grid(X,X_orig,labels,grid_x = 6,grid_y = 6 ):
    s = X.shape
    num_labels = np.max(labels)+1
    img_grid_arr = np.zeros([num_labels,s[1]*grid_x,s[2]*grid_y,3])
    for ind in range(num_labels):
        # print(ind)
        # print(labels)
        # print(np.sum(labels==ind))
        imgs_label = X[labels==ind]
        for i in range(grid_x):
            for j in range(grid_y):
                index = i * grid_y + j -1
                if (j == 0):
                    if(i == 0):
                        row = X_orig[ind]
                    else:
                        row = imgs_label[index]
                else:
                    # print(row.shape)
                    # print(imgs_label[index].shape)
                    row = np.concatenate([row, imgs_label[index]], axis=1)

            if (i == 0):
                img_grid = row
            else:
                img_grid = np.concatenate([img_grid, row], axis=0)

        img_grid_arr[ind] = img_grid
    return img_grid_arr

def image_mean(X,labels,yiq = 0,median = 0 ):
    s = X.shape
    num_labels = np.max(labels)
    img_res_arr = np.zeros([num_labels, s[1], s[2], 3])
    for ind in range(num_labels):
        imgs_label = X[labels == ind]
        if (yiq):
            yiq_imgs = skimage.color.yiq2rgb(imgs_label)
            yiq_med_img = np.nanmedian(yiq_imgs, axis=0)
            res_img = skimage.color.rgb2yiq(yiq_med_img)
            res_img[res_img < 0] = 0
            res_img[res_img > 1] = 1

        else:
            if(median):
                res_img = np.nanmedian(imgs_label, axis=0)

            else:
                res_img = np.nanmean(imgs_label, axis=0)
        img_res_arr[ind] = res_img

    return img_res_arr

def get_shift(img_base,img_cmp,max_shift = 2,img_len=112):
    min_ssd = 100
    min_i = 0
    min_j = 0
    for i in range(-max_shift, max_shift + 1):
        for j in range(-max_shift, max_shift + 1):
            s_x_c = i * (i > 0)
            s_y_c = j * (j > 0)
            e_x_c = i if (i < 0) else img_len
            e_y_c = j if (j < 0) else img_len

            s_x_b = -i * (i < 0)
            s_y_b = -j * (j < 0)
            e_x_b = -i if (i > 0) else img_len
            e_y_b = -j if (j > 0) else img_len
            curr_ssd = np.mean(np.square(img_base[s_x_b:e_x_b, s_y_b:e_y_b] - img_cmp[s_x_c:e_x_c, s_y_c:e_y_c]))
            if (curr_ssd < min_ssd):
                min_ssd = curr_ssd
                min_i = i
                min_j = j
    return [min_i,min_j]


def get_image_aligened(X,labels,images_base,img_len=56):
    res_imgs = np.zeros(images_base.shape)
    for ind in range(max(labels)):
        images_cmp = X[ind == labels]

        img_aligned = np.zeros([images_cmp.shape[0], img_len, img_len, 3])
        img_aligned[:, :, :, :] = np.nan
        for i in range(images_cmp.shape[0]):

            img_base = images_base[ind]
            img_cmp = images_cmp[i]
            shift_x, shift_y = get_shift(img_base, img_cmp,img_len=img_len)

            shift_x = -shift_x
            shift_y = -shift_y
            if (shift_x < 0):
                cmp_x_s = -shift_x
                cmp_x_e = img_len
                arr_x_s = 0
                arr_x_e = img_len + shift_x

            else:
                cmp_x_s = 0
                cmp_x_e = img_len - shift_x
                arr_x_s = shift_x
                arr_x_e = img_len

            if (shift_y < 0):
                cmp_y_s = -shift_y
                cmp_y_e = img_len
                arr_y_s = 0
                arr_y_e = img_len + shift_y

            else:
                cmp_y_s = 0
                cmp_y_e = img_len - shift_y
                arr_y_s = shift_y
                arr_y_e = img_len
            img_aligned[i, arr_x_s:arr_x_e, arr_y_s:arr_y_e] = img_cmp[cmp_x_s:cmp_x_e, cmp_y_s:cmp_y_e]

        res_imgs[ind] = np.nan_to_num(np.nanmean(img_aligned, axis=0))
    return res_imgs


def save_image_arr(images,images_orig = None ,folder=''):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for i in range(images.shape[0]):
        if(images_orig is None):
            scipy.misc.imsave(folder+'img_'+str(i)+'.jpg',images[i])
        else:
            #print(images_orig[i].shape)
            #print(images[i].shape)

            img_concat = np.concatenate([images_orig[i],images[i]],axis=1)
            img_concat = np.squeeze(img_concat)
            #print(img_concat.shape)
            scipy.misc.imsave(folder + 'img_' + str(i) + '.jpg', img_concat)



