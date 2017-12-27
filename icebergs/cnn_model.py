import numpy as np
import pandas as pd
from keras.losses import binary_crossentropy
from keras.optimizers import Adam
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import load_model,Model,Sequential
from keras.callbacks import ModelCheckpoint

from denoising import load_denoised_data

new_model = True

#df = pd.read_json('icebergs/data/train.json')
#band1 = np.array(list(df.band_1.values))
#band2 = np.array(list(df.band_2.values))
#ims1 = np.reshape(band1,(band1.shape[0],75,75))#-band1.min()
#ims2 = np.reshape(band2,(band2.shape[0],75,75))#-band2.min()
#X = np.stack((ims1,ims2),axis=-1)
#X = X[:,10:-10,10:-10]
#print(X.shape)
#meanX = np.mean(X,axis=0)
#stdX  = np.std(X,axis=0)
#X = (X-meanX)/stdX
#Y = df.is_iceberg.values

X,Y = load_denoised_data()

# define parameters
num_classes=1
batch_size = 20
input_shape = X.shape[1:]

if new_model:
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),activation='relu',input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))  #64
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(128, (3, 3), activation='relu'))  #64
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))  #1024
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='sigmoid'))
else:
    old_model = load_model('icebergs/models/my_model_kaggle4.h5')
    intermediate_model = Model(inputs=old_model.input,outputs=old_model.get_layer(index=12).output)
    model=Sequential()
    for layer in intermediate_model.layers:
        layer.trainable=False
        model.add(layer)
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes,activation='sigmoid'))

# data augmentation
datagen = ImageDataGenerator(featurewise_center=False,featurewise_std_normalization=False,rotation_range=180,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True,vertical_flip=True)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
datagen.fit(X_train)

valgen = ImageDataGenerator(featurewise_center=True,featurewise_std_normalization=True)
valgen.fit(X_test)

# train/validate
model.compile(loss=binary_crossentropy,optimizer=Adam(lr=0.0001),metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
if new_model:
    history = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size), steps_per_epoch=int(len(X_train)/batch_size), epochs=1000,validation_data=valgen.flow(X_test,Y_test), verbose=1,callbacks=[checkpointer])
else:
    history = model.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size), steps_per_epoch = int(len(X_train)/batch_size),epochs=300,validation_data=valgen.flow(X_test,Y_test),verbose=1,initial_epoch=150)

model.save('icebergs/models/denoised.h5')
#pickle.dump((meanX,stdX),open('icebergs/models/normalization_kaggle7.pckl','wb'))

#show results
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
