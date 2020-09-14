import numpy as np
from skimage.transform import resize
from scipy.misc import imread,imsave

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



def get_images(dir_, num_images = 50 , resolution = 112):
    arr = np.ones([num_images, resolution, resolution, 3])
    for i in range(num_images):
        file = dir_ + 'img_' + str(i) + '.jpg'
        img = imread(file)
        if(img.shape[0] != resolution):
            img = resize(img, (resolution, resolution), anti_aliasing=True)
        arr[i] = img[:, :]
    return arr

def get_images_kamitani(dir_, num_images = 50 , resolution = 224, index_shift = 0):
    arr = np.ones([num_images, resolution, resolution, 3])
    for i in range(num_images):
        file = dir_ + 'recon_img_S1_VC_Img' + ('0000' + str(i + index_shift))[-4:] + '.jpg'
        img = imread(file)
        arr[i] = img[:, :]
    return arr


def get_images_vim1(dir_, mask_56, num_images = 20 , resolution = 56, index_shift = 0, to_gray = False):
    arr = np.ones([num_images, resolution, resolution, 3])
    for i in range(num_images):
        file = dir_ + 'img_' + str(i) + '.png'
        img = imread(file)

        # if (img.shape[0] != resolution):
        #     img = resize(img, (resolution, resolution), anti_aliasing=True)
        if(to_gray):
            img = img[:, :, :3]
            img = np.multiply(img - 0.5355023, mask_56) + 0.5355023
            img = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
        else:
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)

        arr[i] = img[:, :]
    return arr