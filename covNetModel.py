import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import ops
import keras

#cpu - gpu configuration
config = tf.ConfigProto( device_count = {'GPU': 0 , 'CPU': 256} ) #max: 1 gpu, 56 cpu
sess = tf.Session(config=config) 
keras.backend.set_session(sess)


'''
Second order pooling implementation
'''
# Implementation of LogEig Layer
def _cal_cov_pooling(features):
    #shape_f = features.get_shape().as_list()
    centers_batch = tf.reduce_mean(tf.transpose(features, [0, 2, 1]),2)
    #centers_batch = tf.reshape(centers_batch, [shape_f[0], 1, shape_f[2]])
    centers_batch = tf.reshape(centers_batch, [tf.shape(features)[0], 1, tf.shape(features)[2]])
    centers_batch = tf.tile(centers_batch, [1, tf.shape(features)[1], 1])
    tmp = tf.subtract(features, centers_batch)
    tmp_t = tf.transpose(tmp, [0, 2, 1])
    features_t = 1/tf.cast((tf.shape(features)[1]-1),tf.float32)*tf.matmul(tmp_t, tmp)
    trace_t = tf.trace(features_t)
    trace_t = tf.reshape(trace_t, [tf.shape(features)[0], 1])
    trace_t = tf.tile(trace_t, [1, tf.shape(features)[2]])
    trace_t = 0.0001*tf.matrix_diag(trace_t)
    return tf.add(features_t,trace_t)

# Implementation of LogEig Layer
def _cal_log_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.log(s_f)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t

# computes weights for BiMap Layer
def _variable_with_orth_weight_decay(name1, shape):
    if shape[0] != None:
        s1 = tf.cast(shape[2], tf.int32)
        s2 = tf.cast(shape[2]/2, tf.int32)
        w0_init, _ = tf.qr(tf.random_normal([s1, s2], mean=0.0, stddev=1.0))
        w0 = tf.get_variable(name1, initializer=w0_init)
        tmp1 = tf.reshape(w0, (1, s1, s2))
        tmp2 = tf.reshape(tf.transpose(w0), (1, s2, s1))
        tmp1 = tf.tile(tmp1, [shape[0], 1, 1])
        tmp2 = tf.tile(tmp2, [shape[0], 1, 1])
    else:
        tmp1 = tf.placeholder(tf.float32, shape=(1,None,None))
        tmp2 = tf.placeholder(tf.float32, shape=(1,None,None))
    return tmp1, tmp2

# ReEig Layer
def _cal_rect_cov(features):
    [s_f, v_f] = tf.self_adjoint_eig(features)
    s_f = tf.clip_by_value(s_f, 0.0001, 10000)
    s_f = tf.matrix_diag(s_f)
    features_t = tf.matmul(tf.matmul(v_f, s_f), tf.transpose(v_f, [0, 2, 1]))
    return features_t


def readData():
    with open("/Users/emre/DataIncubatorChallenge2/emotionDetection/kaggleEmotionChallenge/fer2013.csv") as f:
        content = f.readlines()

    lines = np.array(content)
    n = lines.size
    x_train, y_train, x_test, y_test = [], [], [], []
    
    for i in range(1,n):
        try:
            label, img, usage = lines[i].split(",")
         
            val = img.split(" ")
            pixels = np.array(val, 'float32')
         
            label = keras.utils.to_categorical(label, num_classes=7)
         
            if 'Training' in usage:
                y_train.append(label)
                x_train.append(pixels/255)
            elif 'PublicTest' in usage:
                y_test.append(label)
                x_test.append(pixels/255)
        except:
            print('", end="')
        
    #data transformation for train and test sets
    x_train = np.array(x_train, 'float32').reshape(len(x_train), 48, 48, 1)
    y_train = np.array(y_train, 'float32')
    x_test = np.array(x_test, 'float32').reshape(len(x_test), 48, 48, 1)
    y_test = np.array(y_test, 'float32')
    
    return (x_train, y_train, x_test, y_test)

def placeholders(n_H0, n_W0, n_C0, n_y):
    X = tf.placeholder(tf.float32, shape=(None, n_H0, n_W0, n_C0))
    Y = tf.placeholder(tf.float32, shape=(None, n_y))
    return X, Y

def random_mini_batches(X, Y, mini_batch_size = 256, seed = 0):
    
    np.random.seed(seed)            # To make your "random" minibatches the same as ours
    m = X.shape[0]                  # number of training examples
    mini_batches = []
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation,:,:,:]
    shuffled_Y = Y[permutation,:]

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, int(num_complete_minibatches)):
        mini_batch_X = shuffled_X[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:,:,:]
        mini_batch_Y = shuffled_Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[int(num_complete_minibatches) * mini_batch_size : m,:,:,:]
        mini_batch_Y = shuffled_Y[int(num_complete_minibatches) * mini_batch_size : m,:]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def covNetwork(X):

#    # Input Layer
#    input_layer = tf.reshape(features["x"], [-1, 48, 48, 1])
#    
    net = tf.layers.conv2d(X, 64, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='VALID')

    #4
    net = tf.layers.conv2d(net, 96, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='VALID')

    #7
    net = tf.layers.conv2d(net, 128, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.relu(net)
    
    net = tf.layers.conv2d(net, 128, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.relu(net)
    net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='VALID')

    #12
    net = tf.layers.conv2d(net, 256, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.relu(net)

    #14
    net = tf.layers.conv2d(net, 256, kernel_size=[3, 3], padding='SAME')
    net = tf.nn.relu(net)
    
#    shape = net.get_shape().as_list()
#    reshaped = tf.reshape(net, [shape[0], shape[1]*shape[2], shape[3]])
    reshaped = tf.reshape(net, [tf.shape(net)[0], tf.shape(net)[1]*tf.shape(net)[2], tf.shape(net)[3]])
    print reshaped.shape
    local5 = _cal_cov_pooling(reshaped)
    shape = local5.get_shape().as_list()
#    shape = tf.TensorShape(local5.get_shape().as_list())
    weight1, weight2 = _variable_with_orth_weight_decay('orth_weight0', shape)
    local6 = tf.matmul(tf.matmul(weight2, local5), weight1,name='matmulout')
    local7 = _cal_rect_cov(local6)
    local13 = _cal_log_cov(local7)
    if net.get_shape()[1]== None:
        net = tf.reshape(local13,[128,-1])
    else:
        net = tf.reshape(local13,[tf.shape(128), tf.shape(None)])
    
    net = tf.contrib.layers.flatten(net)
    net = tf.contrib.layers.fully_connected(net, 2000, activation_fn=None, reuse=False)
    net = tf.nn.relu(net)
    net = tf.contrib.layers.fully_connected(net, 7, activation_fn=None, reuse=False)
    net = tf.contrib.layers.fully_connected(net, 7, activation_fn=None, 
                weights_initializer=tf.truncated_normal_initializer(stddev=0.1), 
                weights_regularizer=slim.l2_regularizer(0.0005),scope='Logits', reuse=False)

    return net


def main(X_train, Y_train, X_test, Y_test, learning_rate = 0.01,
          num_epochs = 10, minibatch_size = 256, print_cost = True):    
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep results consistent (tensorflow seed)
    seed = 3                                          # to keep results consistent (numpy seed)
    (m, n_H0, n_W0, n_C0) = X_train.shape             
    n_y = Y_train.shape[1]                            
    costs = []                                        # To keep track of the cost
    
    X, Y = placeholders(n_H0, n_W0, n_C0, n_y)
    net = covNetwork(X)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = net, labels = Y))
    
    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer that minimizes the cost.
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    # Initialize all the variables globally
    init = tf.global_variables_initializer()
    
    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
    
        # Run the initialization
        sess.run(init)
    
        # Do the training loop
        for epoch in range(num_epochs):
    
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
            
            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                minibatch_cost += temp_cost / num_minibatches
    
            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)
    
        # Calculate the correct predictions
        predict_op = tf.argmax(net, 1)
        correct_prediction = tf.equal(predict_op, tf.argmax(Y, 1))
        
        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X: X_train, Y: Y_train})
        test_accuracy = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)
                
        return train_accuracy, test_accuracy


x_train, y_train, x_test, y_test = readData()
tran_acc, test_acc = main(x_train, y_train, x_test, y_test)

