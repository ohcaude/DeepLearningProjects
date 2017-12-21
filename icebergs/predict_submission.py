import pickle
import ujson
import numpy as np
from keras.models import load_model

filename = './data/test.json'
with open(filename,'r') as f:
    data = ujson.loads(f.read())

print('data loaded')

ids = [d["id"] for d in data]
band1 = np.array([d["band_1"] for d in data])
band2 = np.array([d["band_2"] for d in data])
del data

X = np.stack((np.reshape(band1,(band1.shape[0],75,75)),np.reshape(band2,(band2.shape[0],75,75))),axis=-1)
del band1
del band2

model = load_model('models/my_model_kaggle5.h5')
meanX,stdX = pickle.load(open('models/normalization_kaggle4.pckl','rb'))

X = (X-meanX) / stdX

y_pred = model.predict(X,verbose=1)

with open('data/submit5.txt','a') as myfile:
    myfile.write("id,is_iceberg\n")
    for idval,prediction in zip(ids,y_pred):
        myfile.write(str(idval)+','+str(prediction[0])+'\n')

