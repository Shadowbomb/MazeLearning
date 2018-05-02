#Python native
from pathlib import Path
#Load Keras
import keras
#Load the Sequential Keras NN model
from keras.models import Sequential
#Load the dropout and flatten libraries and fully connected layer
from keras.layers import Dense, Dropout, Activation, LeakyReLU
#Optimizers
from keras.optimizers import Adam
#Custom activation function and things
from keras.utils.generic_utils import get_custom_objects
from keras.utils import CustomObjectScope
#import model loader
from keras.models import model_from_yaml
#Load Keras backend
from keras import backend as K
#Need tensorflow for custom activation fucntion
import tensorflow as tf
#Numpy for arrays and useful things unlike pandas
import numpy as np
#from matplotlib import pyplot as plt

#Custom imports
from MazeGen import generate
from MazeSolve import solve

#Function for reading mazes: PANDAS SUCKS
def read(file_name,firstLength,nextLength):
    with open(file_name,'r') as scan:
        firstMaze = []
        for i in range(firstLength):
            maze = []
            for j in range(height*width):
                maze.append(float(scan.read(1)))
            scan.readline()
            firstMaze.append(maze)
        nextMaze = []
        for i in range(nextLength):
            maze = []
            for j in range(height*width):
                maze.append(float(scan.read(1)))
            scan.readline()
            nextMaze.append(maze)
    return np.array(firstMaze),np.array(nextMaze)


#Defining custom activation function
# class Threshold(Activation):
#     def __init__(self, activation, **kwargs):
#         super(Threshold, self).__init__(activation, **kwargs)
#         self.__name__ = "Threshold"
#     #Activation function: threshold
#     def threshold(X):
#         X = tf.nn.leaky_relu(X)
#         return tf.cast(X, tf.float32)
#get_custom_objects().update({'Threshold': Threshold(Threshold.threshold)})
def Threshold(X):
    return tf.nn.leaky_relu(X)
get_custom_objects().update({'Threshold': Activation(Threshold)})

#Save model
def save_keras_model(model, model_file_name, weight_file_name):
    model_structure = model.to_yaml()
    with open(model_file_name, "w") as model_file:
        model_file.write(model_structure)
    # serialize weights to HDF5
    model.save_weights(weight_file_name)
    print("Saved model to disk")
#Load Model
def load_keras_model(model_file_name, weight_file_name):
    model_file = open(model_file_name, 'r')
    model_structure = model_file.read()
    model_file.close()
    model = model_from_yaml(model_structure, custom_objects = {'Threshold': Activation(Threshold)})
    # load weights into new model
    model.load_weights(weight_file_name)
    print("Loaded model from disk")
    return model
#Build model
def build_keras_model(io_layer):
    #Initialize and create a Sequential model, it is a linear stack of layers you can pass to contructor to build
    model = Sequential()
    #Fully Connected Layer: Input Layer
    model.add(Dense(io_layer, input_shape = (io_layer,)))
    model.add(Activation(Threshold))
    #Fully Connected Layer: Hidden Layer #1
    model.add(Dense(io_layer * 2))
    model.add(Activation(Threshold))
    model.add(Dropout(0.05))
    #Fully Connected Layer: Hidden Layer #2
    model.add(Dense(io_layer * 3))
    model.add(Activation(Threshold))
    model.add(Dropout(0.07))
    #Fully Connected Layer: Hidden Layer #3
    model.add(Dense(io_layer * 4))
    model.add(Activation(Threshold))
    model.add(Dropout(0.15))
    #Fully Connected Layer: Hidden Layer #4
    model.add(Dense(io_layer * 3))
    model.add(Activation(Threshold))
    model.add(Dropout(0.09))
    #Fully Connected Layer: Hidden Layer #5
    model.add(Dense(io_layer * 2))
    model.add(Activation(Threshold))
    model.add(Dropout(0.001))
    #Fully Connected Layer: Output Layer
    model.add(Dense(io_layer))
    model.add(Activation(Threshold))
    return model

#File names
train_file_name = "SuperMaze.txt"
evaluate_file_name = "SolvedSuperMaze.txt"
predict_file_name = "PredictSuperMaze.txt"
model_file_name = "model.yaml"
weight_file_name = "model.h5"

#Maze width
width = 15
#Maze height
height = 15
#Total number of mazes
maze_total = 150000
#Regenerate and Resolve?
recreate = False

io_layer = width * height
train_percent = 0.95
train_count = int(train_percent * maze_total)
test_count = int(maze_total - train_count)

#Batch processed each step before updating weights, minibatch(batch < dataset && batch > 1)
batch_size = 25
#Number of epochs to train for(epoch = 1 complete iteration over data)
epochs = 1

#RUN IF YOU CHANGE width, height, or maze_total
if (recreate):
    generate(width,height,maze_total,fileName=train_file_name)
    solve(width, height, maze_total, read_file_name = train_file_name, write_file_name = evaluate_file_name)

#Load the data, split between training and testing sets
x_train,x_test = read(train_file_name,train_count,test_count)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
y_train,y_test = read(evaluate_file_name,train_count,test_count)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)

#Keras allows data type specification which can cause speedup over "loose" types in python : float64
#Casting x datasets into defined size
x_train = x_train.astype('float16')
x_test = x_test.astype('float16')
#Casting y datasets into defined size
y_train = y_train.astype('float16')
y_test = y_test.astype('float16')


# load YAML and create model
if Path(model_file_name).exists():
    model = load_keras_model(model_file_name, weight_file_name)
else:
    model = build_keras_model(io_layer)

#Build model, use crossentropy for loss calculation and the Adadelta optimizer for optimizing processing
model.compile(loss = keras.losses.categorical_crossentropy,
    optimizer =
        Adam(lr = 0.001, beta_1 = 0.912, beta_2 = 0.9997, epsilon = None, decay = 0.00001, amsgrad = True),
    metrics = ['accuracy'])
#Fit model to data over ___ epochs with batch size size of ___ and validate against x data and y labels
model.fit(x_train, y_train, batch_size = batch_size, epochs = epochs, verbose = 1, validation_data = (x_test, y_test))
#Evaluate accuracy of model using x test data and y test labels
score = model.evaluate(x_test, y_test, verbose = 1)

#Print results of loss and accuracy
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#Save model
save_keras_model(model, model_file_name, weight_file_name)

result = model.predict(x_test)
result[np.where(result > 0.5)] = 1
result[np.where(result <= 0.5)] = 0
result = result.astype(int)
print(result.shape)
def plotImage(image):
    image = image.reshape(width,height)
    fig = plt.imshow(image)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()
# plotImage(result[0])
np.savetxt(fname=predict_file_name,X=result, fmt="%s",delimiter=" " )