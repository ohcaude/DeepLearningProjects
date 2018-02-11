from keras.applications.inception_v3 import InceptionV3
from keras.models import load_model,Model,Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization

def transfer_inception(feature_layer,input_shape):
    #mixed0: edges  % approx factor of 8 -3, 256 channels
    #mixed4: shapes % approx factor of 16 -2 768 channels
    base_model = InceptionV3(weights='imagenet', include_top=False,input_shape=input_shape)
    input_layer = base_model.layers[0].input
    for layer in base_model.layers:
        layer.trainable = False
        if layer.name == feature_layer:
            feature_layer = layer.output
            #new_model = Model(inputs=[base_model.layers[0].input], outputs=[layer.output])
            break
    return input_layer,feature_layer


def build_model_on_inception(feature_layer,input_shape):
    input_layer,feature_layer = transfer_inception(feature_layer,input_shape)
    
    x = Flatten()(feature_layer)
    x = Dense(64, activation='relu')(x)
    output_layer = Dense(1,activation='sigmoid')(x)

    new_model = Model(inputs=[input_layer],outputs=[output_layer]) 
    return new_model

#model = build_model_on_inception('mixed4',(256,256,3))


#model.summary()
