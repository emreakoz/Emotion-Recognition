from helper_functions import image_from_csv, data_shuffler, test_cv_train_division
from vision_network import network
import pickle

def main():
    #let's first set the hyperparameters and prepare the images for training
    batch_size, epochs, num_classes, input_size, train_split, cv_split, tot_image_size = 128, 30, 8, 128, 0.8, 0.15, 420000
    all_images, all_labels = image_from_csv('pictures_parser/affectNet_filtered.csv',
                                            'pictures_parser/affectNet_images/', input_size, tot_image_size)
    
    #Shuffling and didviding the data into train, cv and test sets
    all_images, all_labels = data_shuffler(all_images, all_labels)
    x_train, x_test, y_train, y_test = test_cv_train_division(all_images, all_labels, train_split, tot_image_size)
    

    model = network(num_classes, input_size)

    fit = True
    if fit == True:
        history = model.fit(x_train, y_train, validation_split = cv_split, epochs=epochs, batch_size=batch_size)
        model_json = model.to_json()
        with open("weights/model.json", "w") as json_file:
            json_file.write(model_json)
        with open('weights/train_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        model.save_weights('weights/model_weights.h5')
    else:
        model.load_weights('weights/model_weights.h5') #load weights
        
    
    test_score = model.evaluate(x_test, y_test, verbose=0)
    probs = model.predict(x_test)
    print('Test Loss:', test_score[0])
    print('Test Accuracy:', 100*test_score[1])
    
    return probs, y_test
    
if __name__ == '__main__':
    probs, y_test = main()