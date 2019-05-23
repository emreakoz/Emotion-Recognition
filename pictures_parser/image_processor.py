import keras
import numpy as np
import pandas as pd
from PIL import Image
import random

tot_image_size, train_image_size, cv_image_size = 300, 1, 2

def marathon_image_from_csv():
    df = pd.read_csv('../london2018.csv').dropna()
#    london2018 = pd.read_csv('../london2018.csv').dropna()
    images = list(df['index'])
    filtered_images = [str(image)+'.png' for image in images]
    labels_from_csv = list(df['emotions'])
    
    all_images = np.empty(shape=(tot_image_size,128,128,3))
    for i,im in enumerate(filtered_images):
        img = Image.open('./filtered_london_pics/'+im)
        all_images[i,:,:,:] = np.array(img)[:,:,:3]
    
    all_labels = [keras.utils.to_categorical(label, 8) for label in labels_from_csv]
    
    return all_images, all_labels

def image_from_csv(path_csv,path_images):
    files = list(pd.read_csv('../'+path_csv).subDirectory_filePath[:tot_image_size])
    filtered_images = [f.split('/')[1] for f in files]
    all_images = np.empty(shape=(tot_image_size,128,128,3))
    for i,im in enumerate(filtered_images):
        img = Image.open('./'+path_images+im)
        all_images[i,:,:,:] = np.array(img)
    
    labels_from_csv = list(pd.read_csv('../'+path_csv).expression[:tot_image_size])
    all_labels = [keras.utils.to_categorical(label, 8) for label in labels_from_csv]
    
    return all_images, all_labels

def data_shuffler(all_images, all_labels):
    shuffler = list(zip(all_images, all_labels))
    random.shuffle(shuffler)
    all_images, all_labels = zip(*shuffler)
    return all_images, all_labels

def test_cv_train_division(all_images, all_labels):
    x_train = np.array(all_images[:train_image_size], 'float32')
    y_train = np.array(all_labels[:train_image_size], 'float32')
    
    x_cv = np.array(all_images[train_image_size+1:train_image_size+cv_image_size], 'float32')
    y_cv = np.array(all_labels[train_image_size+1:train_image_size+cv_image_size], 'float32')
    
    x_test = np.array(all_images[cv_image_size+1:], 'float32')
    y_test = np.array(all_labels[cv_image_size+1:], 'float32')
    
    return x_train, x_cv, x_test, y_train, y_cv, y_test

if __name__ == '__main__':
    all_images, all_labels = marathon_image_from_csv()
#    all_images, all_labels = image_from_csv('denver_filtered.csv','filtered_images/')
    all_images, all_labels = data_shuffler(all_images, all_labels)
    x_train, x_cv, x_test, y_train, y_cv, y_test = test_cv_train_division(all_images, 
                                                                          all_labels)