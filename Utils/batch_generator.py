
from keras.utils import Sequence
from keras.backend import random_normal_variable
import numpy as np
import os
import random
from scipy.misc import imread
import pandas as pd
from Utils.image_functions import image_prepare, rand_shift


class batch_generator_dec(Sequence):

    def __init__(self,  X, Y, batch_size =32, labels = None):
        self.indexes = {}
        self.batch_size = batch_size
        self.Y = Y
        self.X = X
        self.labels = labels
        if self.labels is None:
            self.frac = 1
        else:
            self.frac = (self.labels == 1).sum() #Find the number of fmris per image

        self.indexes  = np.random.permutation(self.Y.shape[0]//self.frac)

    def __len__(self):
        'Denotes the number of batches per epoch'
        if self.batch_size == 12:
            return 75
        len = max(int(self.Y.shape[0] // self.batch_size), 1)
        return len//(self.frac)

    def __getitem__(self,batch_num):
        indexes = (self.indexes)[batch_num * self.batch_size:(batch_num + 1) * self.batch_size]
        if self.frac == 1:
            return self.Y[indexes], self.X[indexes]
        else:
            y = np.zeros((self.batch_size, self.Y.shape[1]))
            x = np.zeros((self.batch_size, self.X.shape[1], self.X.shape[2], self.X.shape[3]))
            for i in range(self.batch_size):
                all_fmris = self.Y[self.labels==indexes[i]]
                num_samples = np.random.randint(3,6)
                samples = np.random.permutation(self.frac)[0:num_samples]  # choose 3 random samples
                chosen_fmris = all_fmris[samples]
                y[i] = np.mean(chosen_fmris, axis=0, keepdims=True)
                x[i] = self.X[self.labels==indexes[i]][0]
            return y, x

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)



class batch_generator_enc(batch_generator_dec):
    def __init__(self, X, Y, batch_size=32,max_shift = 5, labels = None):
        super().__init__(X, Y, batch_size, labels)
        self.max_shift = max_shift

    def __getitem__(self,batch_num, labels = None):
        y, x = super().__getitem__(batch_num)
        x_shifted = np.zeros(x.shape)
        for i in range(x.shape[0]):
            x_shifted[i] = rand_shift(x[i],self.max_shift)
        return x_shifted,y




class batch_generator_external_images(Sequence):
    """
    Gets images from an image directory
    """
    def __init__(self, img_size = 112, batch_size=16,ext_dir = 'data/ImageNet_Files/train_images', num_ext_per_class=150):
        self.img_size = img_size
        self.batch_size = batch_size
        files = os.listdir(ext_dir)
        self.img_files = []
        self.test_im = pd.read_csv('data/Kamitani_Files/images/imageID_test.csv', header=None)

        counter = 0
        flag = False
        self.num_ext_per_class = num_ext_per_class
        for file in files:
            img_files = []
            # Making sure that the chosen class is not a test class
            for i in range(self.test_im[1].shape[0]):
                if file == self.test_im[1][i].split('_')[0]:
                    print(file)
                    flag = True
            if flag:
                flag = False
                continue
            if os.path.isdir(ext_dir + '/' + file) and file.startswith('n'):
                img_files = random.sample(os.listdir(ext_dir + '/' + file), self.num_ext_per_class)
                for i in range(img_files.__len__()):
                    img_files[i] = file + '/' + img_files[i]
                self.img_files += img_files
                counter += 1
            elif file.endswith("JPEG"):
                self.img_files.append(file)
        self.ext_dir = ext_dir
        print('ext_dir: ' + ext_dir + ' img_files len: ' + str(self.img_files.__len__()))
    def __getitem__(self,batch_num):
        img_file = random.sample(self.img_files , self.batch_size)
        images_in = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        images_out = np.zeros([self.batch_size, self.img_size, self.img_size, 3])
        count = 0

        for file in img_file:
            if 'wind' in file:
                img = imread(file)
            else:
                img = imread(self.ext_dir + '/' + file)
            images_in[count] = rand_shift(image_prepare(img, self.img_size), max_shift = 5)
            images_out[count] = image_prepare(img, self.img_size)
            count += 1
        return images_in, images_out

    def __len__(self):
        return  50000// self.batch_size




class batch_generator_test_fmri(Sequence):

    """
    Generates test fMRI samples
    inputs:
        frac - fraction of test fmri to average (3 -> 1/3)
    """
    def __init__(self,Y,labels, batch_size=32, frac =3,ignore_labels = None):
        self.Y = Y
        self.labels = labels
        self.frac = frac
        self.num_vox = Y.shape[1]
        self.batch_size = batch_size
        self.ignore_labels = ignore_labels
        print(self.ignore_labels)

    def __getitem__(self,batch_num):
        y = np.zeros([self.batch_size, self.num_vox])
        for i in range(self.batch_size):
                label = np.random.choice(self.labels, 1)
                if(self.ignore_labels is not None):
                    while(label in self.ignore_labels):
                        label = np.random.choice(self.labels, 1)

                indexes = self.get_random_indexes(label, frac=self.frac)
                y[i] = np.mean(self.Y[indexes, :], axis=0, keepdims=True)

        return y

    def get_random_indexes(self,label,frac =3):
        indexes = np.where(self.labels == label)[0]
        rand_ind = np.random.choice(frac, indexes.shape)
        while (np.sum(rand_ind) == 0 or np.min(rand_ind) > 0):
            rand_ind = np.random.choice(frac, indexes.shape)
        return indexes[rand_ind == 0]#rand_ind


class batch_generator_encdec(Sequence):
    def __init__(self, X, Y, Y_test, test_labels, train_labels = None, batch_paired = 48, batch_unpaired = 16, max_shift_enc = 5, img_len = 112
                 , frac_test = 3,ext_dir = 'data/ImageNet_Files/train_images/', num_ext_per_class=150, ignore_test_fmri_labels = None):
        self.batch_size     = batch_paired+batch_unpaired
        self.gen_dec        = batch_generator_dec(X, Y, batch_size=batch_paired, labels = train_labels)
        self.gen_enc        = batch_generator_enc(X, Y, batch_size=batch_paired,max_shift = max_shift_enc, labels = train_labels)
        self.gen_ext_im     = batch_generator_external_images(img_size = img_len, batch_size=batch_unpaired,ext_dir = ext_dir, num_ext_per_class=num_ext_per_class)
        self.gen_test_fmri  = batch_generator_test_fmri(Y_test,test_labels, batch_size=batch_unpaired, frac =frac_test, ignore_labels = ignore_test_fmri_labels)
        self.batch_paired   = batch_paired
        self.batch_unpaired = batch_unpaired

    def on_epoch_end(self):
        self.gen_dec.on_epoch_end()
        self.gen_enc.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.gen_dec.__len__()
        # return 1
    def __getitem__(self,batch_num):
        y_in, x_out =  self.gen_dec.__getitem__(batch_num)
        x_in, y_out = self.gen_enc.__getitem__(batch_num)
        x_ext_in, x_ext_out = self.gen_ext_im.__getitem__(batch_num)
        y_test_avg = self.gen_test_fmri.__getitem__(batch_num)

        x_in = np.concatenate([x_in, x_ext_in], axis=0)
        x_out = np.concatenate([x_out,x_ext_out], axis=0)
        y_in = np.concatenate([y_in, y_test_avg], axis=0)
        y_out = np.concatenate([y_out, y_test_avg], axis=0)
        mode = np.concatenate([np.ones([self.batch_paired, 1]), np.zeros([self.batch_unpaired, 1])], axis=0)
        return [y_in, x_in, mode], [x_out, y_out]




