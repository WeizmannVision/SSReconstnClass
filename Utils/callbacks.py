import numpy as np
import keras
import scipy.stats as stat
import tensorflow as tf
import scipy
import os
import skimage.color
from Utils.image_loss import *
from scipy.misc import imsave, imread
from Utils.image_functions import image_collage
import io
import matplotlib.pyplot as plt

class log_image_collage_callback(keras.callbacks.Callback):
    def __init__(self, Y, X, model, dir = '',freq = 10):
        self.Y = Y
        self.X = X
        self.pred_model = model
        self.freq = freq
        self.dir = dir

    def on_epoch_end(self, epoch, logs={}):
        if(epoch%self.freq==0):
            X_pred = self.pred_model.predict(self.Y)
            collage = image_collage([self.X,X_pred])
            if not os.path.exists(self.dir):
                os.makedirs(self.dir)


            imsave(self.dir+'ep_'+str(epoch)+'.jpg',collage)




class corr_metric_callback(keras.callbacks.Callback):

    def __init__(self, train_data: object = None, test_data: object = None, generator: object = None, tensorboard_cb: object = None, num_voxels: object = 0,
                 pred_function: object = None,ROI_LVC = None, ROI_HVC = None,encoder_model =None) -> object:
        super().__init__()
        self.train_data = train_data
        self.test_data = test_data
        self.tensorboard_cb=tensorboard_cb
        self.generator = generator
        self.weights =np.ones([1,num_voxels])
        self.num_voxels = num_voxels
        self.pred_function = pred_function
        self.ROI_LVC = ROI_LVC
        self.ROI_HVC = ROI_HVC
        self.encoder_model =encoder_model

    def predict(self,Y):
        if (self.encoder_model is not None):
            return self.encoder_model.predict(Y)
        if(self.pred_function is not None):
            return self.pred_function(self.model,Y)
        else:
            return self.model.predict(Y,batch_size=64)


    def test_top_corr(self,data=None,genetator =None,mode=None, bacth_size=16, num_batches=30, top_k=100, ROI = None):
        s= data[1].shape
        all_samples = s[0]
        test_samples = num_batches*bacth_size
        out_size = s[1]
        indexes = np.random.randint(0, all_samples,test_samples )
        x= (data[0])[indexes]
        y= (data[1])[indexes]
        y_predict = np.zeros([0, out_size ])
        corr = np.zeros([out_size])
        for i in range(num_batches):
            inputs = x[bacth_size*i:bacth_size*(i+1)]
            pred = self.predict(inputs)
            y_predict = np.concatenate((y_predict, pred), axis=0)

        for i in range(out_size):
            corr[i] = stat.pearsonr(y[:, i], y_predict[:, i])[0]
        corr = np.nan_to_num(corr)


        return np.mean(corr),np.median(corr),np.percentile(corr,75),np.percentile(corr,90)



    def weighting(self,corr):
        per_90 = np.percentile(corr, 90)
        per_75 = np.percentile(corr, 75)
        per_50 = np.percentile(corr, 50)
        corr = np.reshape(corr, [1, -1])
        self.weights[corr<per_50] = 0.2
        self.weights[corr > per_75] = 4
        self.weights[corr > per_90] = 10
        self.weights =  self.num_voxels*self.weights/np.sum(self.weights)



    def test_top_corr_gen(self,mode='train', bacth_size=16, num_batches=15,ROI =None):
        test_samples = num_batches*bacth_size
        y_predict = np.zeros([0, self.num_voxels ])
        y_true = np.zeros([0, self.num_voxels])
        corr = np.zeros([self.num_voxels])
        for i in range(num_batches):
            x,y = self.generator.get_batch(mode=mode)
            pred = self.predict(x)
            y_predict = np.concatenate((y_predict, pred), axis=0)
            y_true = np.concatenate((y_true, y), axis=0)



        for i in range(self.num_voxels):
            corr[i] = stat.pearsonr(y_true[:, i], y_predict[:, i])[0]
        corr = np.nan_to_num(corr)
        if(mode == 'test'):
            self.weighting(corr)

        if (ROI is not None):
            corr = corr[ROI]

        return np.mean(corr),np.median(corr),np.percentile(corr,75),np.percentile(corr,90)



    def on_train_begin(self, logs={}):
        self._data = []

    def on_epoch_end(self, epoch, logs={}):
        if(self.generator is not None):
            train_corr_stat = self.test_top_corr_gen(mode='train')
            #val_corr_stat = self.test_top_corr_gen(mode='val')
            test_corr_stat = self.test_top_corr_gen(mode='test')

        else:

            train_corr_stat = self.test_top_corr(self.train_data)
            #val_corr_stat = self.test_top_corr([self.validation_data[0],self.validation_data[1]])
            test_corr_stat = self.test_top_corr(self.test_data)


        print('train:'+str(train_corr_stat))
       # print('val:'+str(val_corr_stat))
        print('test:'+str(test_corr_stat))

        # self._data.append({
        #     'test_corr':test_corr,'val_corr': val_corr,'train_corr':train_corr
        # })

        self.write_log(['test/corr_median','train/corr_median'], [test_corr_stat[1],train_corr_stat[1]], epoch)
        self.write_log(['test/corr_mean','train/corr_mean'], [test_corr_stat[0],train_corr_stat[0]], epoch)
        self.write_log(['test/corr_75', 'train/corr_75'],[test_corr_stat[2],train_corr_stat[2]], epoch)
        self.write_log(['test/corr_90', 'train/corr_90'],[test_corr_stat[3],train_corr_stat[3]], epoch)

        if(self.ROI_HVC is not None):
            train_corr_stat = self.test_top_corr(self.train_data,ROI =self.ROI_HVC)
            test_corr_stat = self.test_top_corr(self.test_data,ROI =self.ROI_HVC)
            self.write_log(['test/corr_hvc_median', 'train/corr_hvc_median'], [test_corr_stat[1], train_corr_stat[1]], epoch)
            self.write_log(['test/corr_hvc_mean', 'train/corr_hvc_mean'], [test_corr_stat[0], train_corr_stat[0]], epoch)
            self.write_log(['test/corr_hvc_75', 'train/corr_hvc_75'], [test_corr_stat[2], train_corr_stat[2]], epoch)
            self.write_log(['test/corr_hvc_90', 'train/corr_hvc_90'], [test_corr_stat[3], train_corr_stat[3]], epoch)



        if (self.ROI_LVC is not None):
            train_corr_stat = self.test_top_corr(self.train_data,ROI =self.ROI_LVC)
            test_corr_stat = self.test_top_corr(self.test_data,ROI =self.ROI_LVC)
            self.write_log(['test/corr_lvc_median', 'train/corr_lvc_median'], [test_corr_stat[1], train_corr_stat[1]], epoch)
            self.write_log(['test/corr_lvc_mean', 'train/corr_lvc_mean'], [test_corr_stat[0], train_corr_stat[0]], epoch)
            self.write_log(['test/corr_lvc_75', 'train/corr_lvc_75'], [test_corr_stat[2], train_corr_stat[2]], epoch)
            self.write_log(['test/corr_lvc_90', 'train/corr_lvc_90'], [test_corr_stat[3], train_corr_stat[3]], epoch)



    def get_data(self):
        return self._data

    def write_log(self,names, logs, batch_no):
        for name, value in zip(names, logs):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = value
            summary_value.tag = name
            self.tensorboard_cb.writer.add_summary(summary, batch_no)
            self.tensorboard_cb.writer.flush()



