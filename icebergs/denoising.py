import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import median
from skimage.morphology import disk

def load_denoised_data():
    df = pd.read_json('icebergs/data/train.json')

    band1 = np.array(list(df.band_1.values))
    band2 = np.array(list(df.band_2.values))
    ims1 = np.reshape(band1,(band1.shape[0],75,75))#-band1.min()
    ims2 = np.reshape(band2,(band2.shape[0],75,75))#-band2.min()
    ims3 = np.divide(ims1,ims2)
    Y = df.is_iceberg.values
    #normalize to visualize
    ims1 = 255*(ims1-np.percentile(ims1,1))/(np.percentile(ims1,99)-np.percentile(ims1,1))
    ims2 = 255*(ims2-np.percentile(ims2,1))/(np.percentile(ims2,99)-np.percentile(ims2,1))
    ims3 = 255*(ims3-np.percentile(ims3,1))/(np.percentile(ims3,99)-np.percentile(ims3,1))

    X = np.stack((ims1,ims2,ims3),axis=-1)

    for c in range(X.shape[0]):
        data = np.squeeze(X[c,:,:,:])
        data[:,:,0] = median(data[:,:,0].astype('uint8'),disk(2))
        data[:,:,1] = median(data[:,:,1].astype('uint8'),disk(2))
        data[:,:,2] = median(data[:,:,2].astype('uint8'),disk(2))
        X[c,:,:,:]=data
        #im = Image.fromarray(np.squeeze(data).astype('uint8'),'RGB')
        #im1 = Image.fromarray(np.squeeze(data[:,:,0]).astype('uint8'))
        #im2 = Image.fromarray(np.squeeze(data[:,:,1]).astype('uint8'))
        #im3 = Image.fromarray(np.squeeze(data[:,:,2]).astype('uint8'))
        #im.show()
        #im1.show()
        #im2.show()
        #im3.show()
        #input()

    X = (X-128)/255
    return X,Y
