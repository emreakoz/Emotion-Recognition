#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import keras
import keras.optimizers as optimizer
from keras.layers import Input,Dense
from keras.models import Model, model_from_json
from helper_functions import freeze_layers

def bilinear_network(num_classes, demographics_size):
    # load json and create model
    json_file = open('weights/model_v7.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    
    vision = model_from_json(loaded_model_json)
    
    # load weights into new model
    vision.load_weights("weights/model_weights_v7.h5")
    vision.layers.pop()
    
    #freezes layers until the given layer
    vision = freeze_layers(vision, 'fc1')
    
    # Initialising the ANN
    demographics_input = Input(shape=(demographics_size,), name= 'dem_input')
    
    # a layer instance is callable on a tensor, and returns a tensor
    demographics_model = Dense(64, activation='relu')(demographics_input)
    demographics_model = Dense(64, activation='relu')(demographics_model)
    demographics_model = Dense(32, activation='relu')(demographics_model)
    
    ## This creates a model that includes
    ## the Input layer and three Dense layers
    demographics = Model(inputs=demographics_input, outputs=demographics_model)
    
    #Merging the vision and demographics networks and adding 
    #another dense layer before the predictions
    merged_model = keras.layers.concatenate([demographics.output, vision.layers[-1].output])
    merged_model = Dense(1024, activation='relu')(merged_model)
    merged_model = Dense(num_classes, activation='sigmoid')(merged_model)
    
    model = Model(inputs=[demographics.input, vision.input], outputs=merged_model)
    model.compile(loss='binary_crossentropy', 
                    optimizer=optimizer.SGD(lr=0.001, momentum=0.9),
                    metrics=['accuracy'])

    return model