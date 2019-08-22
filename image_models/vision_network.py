import tensorflow as tf
import keras
from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy

#num_classes = 8 #angry, disgust, fear, happy, sad, surprise, neutral

def network(num_classes, input_size):
    config = tf.ConfigProto()
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    
    # Initialising the CNN
    model = Sequential()
    
    # 1 - Convolution
    model.add(Conv2D(96,(8,8), strides=(2, 2), padding='valid', input_shape=(input_size, input_size, 3), name = 'conv_layer1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), name = 'pool_layer1'))
#    model.add(Dropout(0.25))
    
    # 2nd Convolution layer
    model.add(Conv2D(256,(4,4), padding='valid', name = 'conv_layer2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), name = 'pool_layer2'))
#    model.add(Dropout(0.25))
    
    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), padding='same', name = 'conv_layer3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    # 4th Convolution layer
    model.add(Conv2D(512,(3,3), padding='same', name = 'conv_layer4'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
#    model.add(MaxPooling2D(pool_size=(2, 2)))
#    model.add(Dropout(0.25))
    
    # 5th Convolution layer
    model.add(Conv2D(512,(3,3), padding='same', name = 'conv_layer5'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), name = 'pool_layer3'))
#    model.add(Dropout(0.25))
    
    # Flattening
    model.add(Flatten())
    
    # Fully connected layer 1st layer
    model.add(Dense(4096, kernel_regularizer=regularizers.l2(0.001), name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    
    # Fully connected layer 2nd layer
    model.add(Dense(2048, kernel_regularizer=regularizers.l2(0.001), name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    # Fully connected layer 3nd layer
    model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.001), name='fc3'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='sigmoid', name='predictions'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                  metrics=[categorical_accuracy])
    return model