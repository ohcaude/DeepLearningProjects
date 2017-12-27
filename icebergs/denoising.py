import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

df = pd.read_json('icebergs/data/train.json')

band1 = np.array(list(df.band_1.values))
band2 = np.array(list(df.band_2.values))
ims1 = np.reshape(band1,(band1.shape[0],75,75))#-band1.min()
ims2 = np.reshape(band2,(band2.shape[0],75,75))#-band2.min()
ims3 = np.divide(ims1,ims2)

#normalize to visualize
ims1 = 255*(ims1-ims1.min())/(ims1.max()-ims1.min())
#ims1 = 255*(1-ims1)
#ims1 = ims1/np.percentile(ims1,90)
ims2 = 255*(ims2-ims2.min())/(ims2.max()-ims2.min())
#ims2 = ims2/np.percentile(ims2,90)
#ims3 = (ims3-ims3.min())/(ims3.max()-ims3.min())
#ims3 = 255*(1-ims3)
ims3 = 255*(ims3-np.percentile(ims3,1))/(np.percentile(ims3,99)-np.percentile(ims3,1))

X = np.stack((ims1,ims2,ims3),axis=-1)

for c in range(X.shape[0]):
    im = Image.fromarray(np.squeeze(X[c,:,:,:]).astype('uint8'),'RGB')
    #im1 = Image.fromarray(np.squeeze(X[c,:,:,0]).astype('uint8'))
    #im2 = Image.fromarray(np.squeeze(X[c,:,:,1]).astype('uint8'))
    #im3 = Image.fromarray(np.squeeze(X[c,:,:,2]).astype('uint8'))

    plt.figure()
    plt.imshow(im)
    #plt.figure()
    #plt.imshow(im1)
    #plt.figure()
    #plt.imshow(im2)
    #plt.figure()
    #plt.imshow(im3)
    plt.show()

