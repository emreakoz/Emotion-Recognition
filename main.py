from image_processor import image_from_csv, data_shuffler, test_cv_train_division, marathon_image_from_csv
from network import network
from keras.preprocessing.image import ImageDataGenerator

def main():
    batch_size, steps, epochs = 64, 150, 25
    all_images, all_labels = marathon_image_from_csv()
#    all_images, all_labels = image_from_csv('denver_filtered.csv','filtered_images/')
    all_images, all_labels = data_shuffler(all_images, all_labels)
    x_train, x_cv, x_test, y_train, y_cv, y_test = test_cv_train_division(all_images, 
                                                                          all_labels)
    model = network()

    fit = True
    if fit == True:
        model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        model.save_weights('./model_weights.h5')
    else:
        model.load_weights('./model_weights.h5') #load weights

#    train_score = model.evaluate(x_train, y_train, verbose=0)
#    print('Train loss:', train_score[0])
#    print('Train accuracy:', 100*train_score[1])
     
#    cv_score = model.evaluate(x_cv, y_cv, verbose=0)
#    print('Cross Validation Loss:', cv_score[0])
#    print('Cross Validation Accuracy:', 100*cv_score[1])
    
    test_score = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test)
    print('Test Loss:', test_score[0])
    print('Test Accuracy:', 100*test_score[1])
    
    return probs, y_test
    
if __name__ == '__main__':
    probs, y_test = main()

