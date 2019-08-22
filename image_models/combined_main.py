from helper_functions import data_shuffler, test_cv_train_division, \
                            marathon_image_from_csv, get_demographics
                            
from combined_network import bilinear_network
import pickle

def bilinear_main():
    #let's first set the hyperparameters and prepare the 
    #images and demographics information for training
    batch_size, epochs, num_classes, input_size, train_split, cv_split, tot_data_size = 128, 10, 8, 128, 0.9, 0.1, 601
    all_images, all_labels = marathon_image_from_csv(input_size, tot_data_size)
    demographics_x = get_demographics()
    
    
    #Shuffling and didviding the data into train, cv and test sets
    all_images, all_labels, demographics_x = data_shuffler(all_images, all_labels, demographics_x)
    x_vision_train, x_vision_test, y_train, y_test = test_cv_train_division(all_images, all_labels, train_split, tot_data_size)
    x_dem_train, x_dem_test, y_train, y_test = test_cv_train_division(demographics_x, all_labels, train_split, tot_data_size)
    
    
    #call the bilinear network for fine tuning the vision 
    #network and training the ann for demographics network
    model = bilinear_network(num_classes, len(demographics_x[0]))
    
    fit = True
    if fit == True:
        history = model.fit([x_dem_train, x_vision_train], y_train, validation_split = cv_split, epochs=epochs, batch_size=batch_size)
        model_json = model.to_json()
        with open("weights/combined_model.json", "w") as json_file:
            json_file.write(model_json)
        with open('weights/combined_train_history', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        model.save_weights('weights/combined_model.h5')
    else:
        model.load_weights('weights/combined_model.h5') #load weights

    test_score = model.evaluate([x_dem_test, x_vision_test], y_test, verbose=0)
    probs = model.predict([x_dem_test, x_vision_test])
    print('Test Loss:', test_score[0])
    print('Test Accuracy:', 100*test_score[1])
    
    return probs, y_test
    
if __name__ == '__main__':
    probs, y_test = bilinear_main()