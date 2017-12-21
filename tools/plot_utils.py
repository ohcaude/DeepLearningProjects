import matplotlib.pyplot as plt
import numpy as np

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

