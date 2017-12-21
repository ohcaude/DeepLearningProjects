import matplotlib.pyplot as plt
import numpy as np
import pickle
from keras import backend as K
from keras.models import load_model,Model

model = load_model('my_model_kaggle5.h5')

inp = model.input

print(model.summary())

import pandas as pd
df = pd.read_pickle('features.pckl')
band1 = np.array(list(df.band_1.values))
band2 = np.array(list(df.band_2.values))
ims1 = np.reshape(band1,(band1.shape[0],75,75))#-band1.min()
ims2 = np.reshape(band2,(band2.shape[0],75,75))#-band2.min()

X = np.stack((ims1,ims2),axis=-1)
meanX,stdX = pickle.load(open('normalization_kaggle5.pckl','rb'))
X = (X-meanX) / stdX

intermediate_model = Model(inputs=model.input,outputs=model.get_layer(index=12).output)
F = intermediate_model.predict(X,verbose=1)
Ypred = model.predict(X,verbose=1)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
Xpca = PCA(n_components=50).fit_transform(F)
Xtsne = TSNE(n_components=2).fit_transform(Xpca)

labels = df['is_iceberg'].values

plt.figure()
plt.scatter(Xtsne[np.where(labels==1),0],Xtsne[np.where(labels==1),1])
plt.scatter(Xtsne[np.where(labels==0),0],Xtsne[np.where(labels==0),1])
plt.figure()
sc=plt.scatter(Xtsne[:,0],Xtsne[:,1],c=Ypred,cmap=plt.cm.Spectral)
plt.colorbar(sc)
plt.show()
