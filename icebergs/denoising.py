import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.filters import gaussian

def load_denoised_data():
    df = pd.read_json('icebergs/data/train.json')

    band1 = np.array(list(df.band_1.values))
    band2 = np.array(list(df.band_2.values))
    ims1 = np.reshape(band1,(band1.shape[0],75,75))#-band1.min()
    ims2 = np.reshape(band2,(band2.shape[0],75,75))#-band2.min()
    ims3 = np.divide(ims1,ims2)
    Y = df.is_iceberg.values
    #normalize to visualize
    ims1 = (ims1-np.percentile(ims1,1))/(np.percentile(ims1,99)-np.percentile(ims1,1))
    ims2 = (ims2-np.percentile(ims2,1))/(np.percentile(ims2,99)-np.percentile(ims2,1))
    ims3 = (ims3-np.percentile(ims3,1))/(np.percentile(ims3,99)-np.percentile(ims3,1))

    X = np.stack((ims1,ims2,ims3),axis=-1)

    for c in range(X.shape[0]):
        data = np.squeeze(X[c,:,:,:])
        X[c,:,:,:]= gaussian(data,sigma=1,multichannel=True)
        #im = Image.fromarray(np.squeeze(255*X[c,:,:,:]).astype('uint8'),'RGB')
        #im1 = Image.fromarray(np.squeeze(255*X[c,:,:,0]).astype('uint8'))
        #im2 = Image.fromarray(np.squeeze(255*X[c,:,:,1]).astype('uint8'))
        #im3 = Image.fromarray(np.squeeze(255*X[c,:,:,2]).astype('uint8'))
        #im.show()
        #im1.show()
        #im2.show()
        #im3.show()
        #input()

    #print(np.mean(X,axis=0))
    #print(np.std(X,axis=0))
    X = (X-np.mean(X,axis=0))/np.std(X,axis=0)
    return X,Y


