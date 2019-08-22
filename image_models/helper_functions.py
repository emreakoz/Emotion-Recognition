import keras
import numpy as np
import pandas as pd
from PIL import Image
import random


def freeze_layers(model, layer_name):
    trainable = False
    for layer in model.layers:
        if layer.name == layer_name:
            trainable = True
        layer.trainable = trainable
    return model


def standard_scaler(data):
    mean, std = data.mean(), data.std()
    data = (data - mean) / std
    return data


def get_demographics():
    chicago2018 = pd.read_csv('../marathon_demographics/chicago2018_model.csv').dropna().reset_index()
    london2018 = pd.read_csv('../marathon_demographics/london2018.csv').dropna().reset_index()
    df = pd.concat([chicago2018,london2018]).reset_index(drop=True)
    
    df = df.replace('M', 0)
    df = df.replace('F', 1)
    
    df = df.replace('18-39', 0)
    df = df.replace('40-44', 1)
    df = df.replace('45-49', 2)
    df = df.replace('50-54', 3)
    df = df.replace('55-59', 4)
    df = df.replace('60-64', 5)
    df = df.replace('65-69', 6)
    df = df.replace('70+', 7)
    
    df = df.replace('YES', 0)
    df = df.replace('NO', 1)
    
    
    df['gender_rankings'] = standard_scaler(df['gender_rankings'])
    
    genders, bq, ages = [], [], []
    
    for i,gender in enumerate(df['index']):
        genders.append(keras.utils.to_categorical(df['gender'][i], 2))
        ages.append(keras.utils.to_categorical(df['ages'][i], 8))
        bq.append(keras.utils.to_categorical(df['BQ'][i], 2))
      
    demographics_X = [list(ages[i])+list(genders[i])+list(bq[i])+[df['gender_rankings'][i]] for i in range(len(genders))]

    return demographics_X


def marathon_image_from_csv(input_size, tot_image_num):
    chicago2018 = pd.read_csv('../marathon_demographics/chicago2018_model.csv').dropna().reset_index()
    london2018 = pd.read_csv('../marathon_demographics/london2018.csv').dropna().reset_index()
    df = pd.concat([chicago2018,london2018]).reset_index(drop=True)
    
    all_images = np.empty(shape=(tot_image_num,input_size,input_size,3))
    for i,im in enumerate(chicago2018['index']):
        img = Image.open('../pictures_parser/filtered_chicago_pics/'+ str(chicago2018['index'][i]) + '.png')
        all_images[i,:,:,:] = np.array(img)[:,:,:3]
        
    for j,im in enumerate(london2018['index']):
        img = Image.open('../pictures_parser/filtered_london_pics/'+ str(london2018['index'][j]) + '.png')
        all_images[i+j,:,:,:] = np.array(img)[:,:,:3]

    labels_from_csv = list(df['emotions'])
    all_labels = [keras.utils.to_categorical(label, 8) for label in labels_from_csv]
    
    return all_images, all_labels


def image_from_csv(path_csv,path_images, input_size, tot_image_num):
    files = list(pd.read_csv('../'+path_csv).subDirectory_filePath[:tot_image_num])
    filtered_images = [f.split('/')[1] for f in files]
    all_images = np.empty(shape=(len(files),input_size,input_size,3))
    for i,im in enumerate(filtered_images):
        img = Image.open('../'+path_images+im)
        all_images[i,:,:,:] = np.array(img)
    
    labels_from_csv = list(pd.read_csv('../'+path_csv).expression[:tot_image_num])
    all_labels = [keras.utils.to_categorical(label, 8) for label in labels_from_csv]
    
    return all_images, all_labels


def data_shuffler(all_images, all_labels, demographics_x = []):
    if demographics_x:
        shuffler = list(zip(all_images, all_labels, demographics_x))
        random.shuffle(shuffler)
        all_images, all_labels, demographics_x = zip(*shuffler)
        return all_images, all_labels, demographics_x
    else:
        shuffler = list(zip(all_images, all_labels))
        random.shuffle(shuffler)
        all_images, all_labels = zip(*shuffler)
        return all_images, all_labels


def test_cv_train_division(all_images, all_labels, train_split, tot_image_num):
    train_image_size = int(train_split * tot_image_num)
    x_train = np.array(all_images[:train_image_size], 'float32')
    y_train = np.array(all_labels[:train_image_size], 'float32')
    
    x_test = np.array(all_images[train_image_size+1:], 'float32')
    y_test = np.array(all_labels[train_image_size+1:], 'float32')
    
    return x_train, x_test, y_train, y_test