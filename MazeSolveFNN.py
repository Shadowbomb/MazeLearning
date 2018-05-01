#Imports to use Keras for CNN
from random import random
#Load Keras
import keras
#Load the Sequential Keras NN model
from keras.models import Sequential
#Load the dropout and flatten libraries and fully connected layer
from keras.layers import Dense, Dropout, Activation
#Custom activation function
from keras.utils.generic_utils import get_custom_objects
#Load Keras backend
from keras import backend as K
#Need tensorflow for custom activation fucntion
import tensorflow as tf
#Numpy for arrays and useful things unlike pandas
import numpy as np

#Maze information
width = 64
height = 64
maze_total = 1000

io_layer = width * height
train_file_name = "SuperMaze.txt"
evaluate_file_name = "SolvedSuperMaze.txt"
predict_file_name = "PredictSuperMaze.txt"
train_percent = 0.9
train_count = int(train_percent * maze_total)
test_count = int(maze_total - train_count)

#Batch processed each step before updating weights, minibatch(batch < dataset && batch > 1)
batch_size = 10
#Number of epochs to train for(epoch = 1 complete iteration over data)
epochs = 1

#Load the data, split between training and testing sets from MNIST(half of MNIST for training and half for testing)
def read(file_name):
    with open(file_name,'r') as scan:
        allMaze = []
        for i in range(maze_total):
            maze = []
            for j in range(height*width):
                maze.append(float(scan.read(1)))
            scan.readline()
            allMaze.append(maze)
    return allMaze

unsolved_data = np.array(read(train_file_name))
x_train = unsolved_data[0:train_count:]
x_test = unsolved_data[train_count:]
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

solved_data = np.array(read(evaluate_file_name))
y_train = solved_data[0:train_count:]
y_test = solved_data[train_count:]
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#Keras allows data type specification which can cause speedup over "loose" types in python : float64
#Casting x datasets into defined size
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')
#Casting y datasets into defined size
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')

#Defining custom activation function
def Threshold(x):
    cond = tf.greater_equal(x, tf.constant(0.5))
    return tf.where(cond, tf.constant(1.0), tf.constant(0.0))
get_custom_objects().update({'Threshold': Activation(Threshold)})

#Initialize and create a Sequential model, it is a linear stack of layers you can pass to contructor to build
model = Sequential()
#Fully Connected Layer 1, uses io_layer nodes and ReLu function for activation, outputs io_layer nodes
model.add(Dense(io_layer, input_shape = (io_layer,), activation = 'sigmoid'))
model.add(Dense(io_layer, activation = 'sigmoid'))
#Fully Connected Layer 2, output layer which contains total number of outputs(classes) and softmax activation function
model.add(Dense(io_layer, activation = Threshold))

#Build model, use crossentropy for loss calculation and the Adadelta optimizer for optimizing processing
model.compile(loss = keras.losses.categorical_crossentropy, optimizer = keras.optimizers.Adadelta(), metrics = ['accuracy'])

#Fit model to data over ___ epochs with batch size size of ___ and validate against x data and y labels
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
#Evaluate accuracy of model using x test data and y test labels
score = model.evaluate(x_test, y_test, verbose = 0)

#Print results of loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

mnum = int(random() * maze_total)
print("Maze number:", mnum)
#x_prove = x_test[mnum:mnum]
result = model.predict(x = x_test, batch_size = 1, verbose = 0)
print(result)
np.savetxt(fname=predict_file_name, )