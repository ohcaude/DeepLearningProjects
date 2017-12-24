import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


df = pd.read_json('icebergs/data/train.json')

band1i = np.array(list(df[df['is_iceberg']==1].band_1.values))
band2i = np.array(list(df[df['is_iceberg']==1].band_2.values))
template1i = np.reshape(band1i,(band1i.shape[0],75,75))
template1i = np.squeeze(np.mean(template1i,axis=0))
template2i = np.reshape(band2i,(band2i.shape[0],75,75))
template2i = np.squeeze(np.mean(template2i,axis=0))
template1i = 255*(template1i-template1i.min())/(template1i.max()-template1i.min())
template2i = 255*(template2i-template2i.min())/(template2i.max()-template2i.min())

band1b = np.array(list(df[df['is_iceberg']==0].band_1.values))
band2b = np.array(list(df[df['is_iceberg']==0].band_2.values))
template1b = np.reshape(band1b,(band1b.shape[0],75,75))
template1b = np.squeeze(np.mean(template1b,axis=0))
template2b = np.reshape(band2b,(band2b.shape[0],75,75))
template2b = np.squeeze(np.mean(template2b,axis=0))
template1b = 255*(template1b-template1b.min())/(template1b.max()-template1b.min())
template2b = 255*(template2b-template2b.min())/(template2b.max()-template2b.min())

plt.figure()
plt.imshow(template1i.astype('uint8'))
plt.draw()
plt.figure()
plt.imshow(template2i.astype('uint8'))
plt.draw()
plt.figure()
plt.imshow(template1b.astype('uint8'))
plt.draw()
plt.figure()
plt.imshow(template2b.astype('uint8'))
plt.show()

#band1.sort(axis=1)
#band2.sort(axis=1)

#band1_brilliance = np.mean(band1[:,-300:],axis=1)
#band2_brilliance = np.mean(band2[:,-300:],axis=1)

#df['band1_brilliance']=pd.Series(band1_brilliance)
#df['band2_brilliance']=pd.Series(band2_brilliance)

#df.plot.scatter('band1_brilliance','band2_brilliance',c='is_iceberg',colormap='jet')
#plt.show()

