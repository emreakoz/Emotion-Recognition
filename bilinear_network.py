import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input,Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from keras.metrics import categorical_accuracy
from keras.layers.normalization import BatchNormalization


num_classes = 8 #angry, disgust, fear, happy, sad, surprise, neutral

def bilinear_network():

    config = tf.ConfigProto()
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)
    
    # Initialising the CNN
    vision_input = Input(shape=(128, 128, 3))
    #vision_model = Sequential()
    
    # 1 - Convolution
    vision_model = Conv2D(64, (3, 3), padding='same', activation='relu')(vision_input)
    vision_model = MaxPooling2D(pool_size=(2, 2))(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Dropout(0.25)(vision_model)
    
    
    
    # 2nd Convolution layer
    vision_model = Conv2D(128, (5, 5), padding='same', activation='relu')(vision_model)
    vision_model = MaxPooling2D(pool_size=(2, 2))(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Dropout(0.25)(vision_model)
    
    # 3rd Convolution layer
    vision_model = Conv2D(512, (3, 3), padding='same', activation='relu')(vision_model)
    vision_model = MaxPooling2D(pool_size=(2, 2))(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Dropout(0.25)(vision_model)
    
    
    # 4th Convolution layer
    vision_model = Conv2D(512, (3, 3), padding='same', activation='relu')(vision_model)
    vision_model = MaxPooling2D(pool_size=(2, 2))(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Dropout(0.25)(vision_model)
    
    # Flattening
    vision_model = Flatten()(vision_model)
    
    
    # Fully connected layer 1st layer
    vision_model = Dense(256, activation='relu')(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Dropout(0.25)(vision_model)
    
    # Fully connected layer 2nd layer
    vision_model = Dense(512, activation='relu')(vision_model)
    vision_model = BatchNormalization()(vision_model)
    vision_model = Dropout(0.25)(vision_model)
    
    
    ## Now let's get a tensor with the output of our vision model:
    vision = Model(inputs=vision_input, outputs=vision_model)
    
    
    
    # Initialising the ANN
    demographics_input = Input(shape=(13,))
    
    # a layer instance is callable on a tensor, and returns a tensor
    demographics_model = Dense(64, activation='relu')(demographics_input)
    demographics_model = Dense(64, activation='relu')(demographics_model)
    demographics_model = Dense(32, activation='relu')(demographics_model)
    #predictions = Dense(8, activation='softmax')(x)
    
    ## This creates a model that includes
    ## the Input layer and three Dense layers
    demographics = Model(inputs=demographics_input, outputs=demographics_model)
    
    
    
    
    merged_model = keras.layers.concatenate([demographics.output, vision.output])
    merged_model = Dense(1024, activation='relu')(merged_model)
    merged_model = Dense(num_classes, activation='softmax')(merged_model)
    
    model = Model(inputs=[demographics.input, vision.input], outputs=merged_model)
    model.compile(optimizer='adam', loss='binary_crossentropy', 
                         metrics=[categorical_accuracy])
    
    return model

if __name__ == '__main__':
    model = bilinear_network()