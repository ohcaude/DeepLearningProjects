import pickle
import ujson
import numpy as np
from keras.models import load_model
import subprocess
from denoising import load_denoised_data
#filename = './data/test.json'
#with open(filename,'r') as f:
#    data = ujson.loads(f.read())

#print('data loaded')

#ids = [d["id"] for d in data]
#band1 = np.array([d["band_1"] for d in data])
#band2 = np.array([d["band_2"] for d in data])
#band3 = band1/band2
#del data

#X = np.stack((np.reshape(band1,(band1.shape[0],75,75)),np.reshape(band2,(band2.shape[0],75,75)),np.reshape(band3,(band3.shape[0],75,75))),axis=-1)
#del band1
#del band2
X,ids = load_denoised_data(True)

model = load_model('icebergs/models/denoised3.h5')
#meanX,stdX = pickle.load(open('models/normalization_kaggle8.pckl','rb'))

#X = (X-meanX) / stdX

y_pred = model.predict(X,verbose=1)

with open('icebergs/data/submit11.txt','a') as myfile:
    myfile.write("id,is_iceberg\n")
    for idval,prediction in zip(ids,y_pred):
        myfile.write(str(idval)+','+str(prediction[0])+'\n')
print(model.summary())
fname='test.json'
username='ocaudevi'
password='eO7MYdZ4Kni5'
competition='statoil-iceberg-classifier-challenge'
subprocess.run(['kg', 'submit','icebergs/data/submit11.txt', '-u', username, '-p', password ,'-c', competition])
