import numpy as np
from skimage.io import imread
from transfer_learning import transfer_inception 
from skimage.transform import resize
from keras.models import load_model,Model,Sequential

example_file = 'DSC_0012.JPG'
im = imread(example_file)
im = im[1250:1750,2500:3000,:]
#im = resize(im,output_shape=(150,300,3))

import matplotlib.pyplot as plt

plt.figure()
plt.imshow(im)

im = np.expand_dims(im,axis=0)

input_layer,output_layer = transfer_inception('mixed0',im.shape[1:])
model = Model(inputs=[input_layer],outputs=[output_layer])
features = model.predict(im,verbose=1)

print(features.shape)

plt.figure()
plt.imshow(features[0,:,:,0])


plt.show()
