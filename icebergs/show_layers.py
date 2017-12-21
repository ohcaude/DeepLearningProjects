import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.insert(0,'../tools')
import plot_tools

def plot_layer(layer_output):
    layer_output = np.squeeze(layer_output)
    nrows = int(np.ceil(np.sqrt(layer_output.shape[-1])))
    nfilt = layer_output.shape[-1]
    print(layer_output.shape)
    mosaic = np.zeros((layer_output.shape[0]*nrows,layer_output.shape[1]*nrows))
    for col in range(nrows):
        data = layer_output[:,:,(col*nrows):((col+1)*nrows)]
        print(data.shape)
        data = np.swapaxes(data,0,2)
        data = np.reshape(data,(-1,layer_output.shape[1]))
        mosaic[0:data.shape[0],(col*layer_output.shape[1]):((col+1)*layer_output.shape[1])] = data
    mosaic = np.array(255*(mosaic-mosaic.min())/(mosaic.max()-mosaic.min()))
    plt.figure();
    plt.imshow(mosaic.astype('uint8'))




from keras import backend as K
from keras.models import load_model

model = load_model('my_model_kaggle4.h5')

inp = model.input
outputs = [layer.output for layer in model.layers]
functors = [K.function([inp]+ [K.learning_phase()],[out]) for out in outputs]

#test = np.random.random((1,75,75,2))
#layer_outs = [func([test,1.]) for func in functors]
#plot_layer(layer_outs[0][0])
#for l in layer_outs[0:11]:
#    plot_layer(l[0])
    

import pandas as pd

df = pd.read_pickle('features.pckl')

band1 = np.array(list(df.band_1.values))
band2 = np.array(list(df.band_2.values))
ims1 = np.reshape(band1,(band1.shape[0],75,75))#-band1.min()
ims2 = np.reshape(band2,(band2.shape[0],75,75))#-band2.min()
X = np.stack((ims1,ims2),axis=-1)

np.random.shuffle(X)

for s in range(X.shape[0]):
    layer_outs = [func([np.expand_dims(X[s,:,:,:],axis=0),1.]) for func in functors]
    plot_layer(layer_outs[0][0])
    plt.show(block=True)
