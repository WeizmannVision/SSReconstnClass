import numpy as np
import scipy.stats as stat


def calc_snr(y, y_avg, labels):
    sig = np.var(y_avg, axis=0)
    noise = 0
    for l in labels:
        noise += np.var(y[labels == l], axis=0)
    noise /= len(labels)
    return sig/noise



def corr_percintiles(y,y_pred, per = [50,75,90]):
    num_voxels = y.shape[1]
    corr = np.zeros([num_voxels])

    for i in range(num_voxels):
        corr[i] = stat.pearsonr(y[:, i], y_pred[:, i])[0]
    corr = np.nan_to_num(corr)

    corr_per = []
    for p in per:
        corr_per.append(np.percentile(corr,p))
    return corr_per