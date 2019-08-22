import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Activation,Dense,Dropout,Flatten
from keras.metrics import categorical_accuracy
from keras.layers.normalization import BatchNormalization


def network(num_classes, input_size):
    config = tf.ConfigProto()
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    
    # Initialising the CNN
    vision_input = Input(shape=(input_size, input_size, 3))
    
    # 1 - Convolution
    vision_model = Conv2D(96, (8, 8), padding='valid', name = 'conv_layer1')(vision_input)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    vision_model = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name = 'pool_layer1')(vision_model)
#    vision_model = Dropout(0.25)(vision_model)
    
    # 2nd Convolution layer
    vision_model = Conv2D(256, (4, 4), padding='valid', name = 'conv_layer2')(vision_input)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    vision_model = MaxPooling2D(pool_size=(3, 3), strides=(2,2), name = 'pool_layer2')(vision_model)

    # 3rd Convolution layer
    vision_model = Conv2D(512, (3, 3), padding='valid', name = 'conv_layer3')(vision_input)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    
    # 4th Convolution layer
    vision_model = Conv2D(512, (3, 3), padding='valid', name = 'conv_layer4')(vision_input)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    
    # 5th Convolution layer
    vision_model = Conv2D(512, (3, 3), padding='valid', name = 'conv_layer5')(vision_input)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    vision_model = MaxPooling2D(pool_size=(2, 2), strides=(2,2), name = 'pool_layer3')(vision_input)
    
    # Flattening
    vision_model = Flatten()(vision_model)
    
    # Fully connected layer 1st layer
    vision_model = Dense(4096, name='fc1')(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    vision_model = Dropout(0.4)(vision_model)
    
    # Fully connected layer 2nd layer
    vision_model = Dense(2048, name='fc2')(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    vision_model = Dropout(0.4)(vision_model)
    
    # Fully connected layer 3rd layer
    vision_model = Dense(1024, name='fc3')(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Activation('relu')(vision_model)
    vision_model = Dropout(0.4)(vision_model)

    # Prediction layer
    predictions = Dense(num_classes, activation='sigmoid', name='predictions')(vision_model)
    
    ## Now let's get a tensor with the output of our vision model:
    vision = Model(inputs=vision_input, outputs=predictions)
    vision.compile(optimizer='adam', loss='binary_crossentropy', 
                         metrics=[categorical_accuracy])
    
    return vision
