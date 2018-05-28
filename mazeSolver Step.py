from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from MazeGen import generate
from MazeSolve import solve

recreate = True

x_file_name = "SuperMaze.txt"
y_file_name = "SolvedSuperMaze.txt"
model_file_name = "model.yaml"
weight_file_name = "model.h5"

width = 10
height = 10
maze_total = 250
train_percent = 0.95
train_count = int(train_percent * maze_total)
test_count = int(maze_total - train_count)

directions = [[1,0],[0,1],[-1,0],[0,-1]]
get_direction = [[0,2],[3,0,1],[0,0]]

io_layer = width * height - 1
batch_size = 1
epoch = 15
run_per_epoch = 0.2 * width*height

if (recreate):
    print("Start Generation for: " + str(maze_total) + " " + str(width) + "x" + str(height) + " maze(s)")
    generate(width,height,maze_total,fileName=x_file_name)
    solve(width, height, maze_total, read_file_name = x_file_name, write_file_name = y_file_name)

model = Sequential()
model.add(Dense(io_layer, input_shape = (io_layer,)))
model.add(Dropout(0.25))
model.add(Dense(10,activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(4,activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy,
    optimizer = Adam(lr = 1e-3),metrics = ['accuracy'])

start_x = 1
start_y = 1

end_x = width-2
end_y = height-2

def spiralMaze(grid,start_x,start_y):
    array = []
    height = len(grid)
    width = len(grid[0])
    distance = max(max(start_x+1,abs(start_x-width)),max(start_y+1,abs(start_y-height)))
    previous_x = -1000
    previous_y = -1000
    for i in range(1,distance):
        upper_limit_x = start_x + i
        upper_limit_y = start_y + i
        lower_limit_x = start_x - i
        lower_limit_y = start_y - i

        for side in range(4):
            x = lower_limit_x if side == 0 or side == 3 else upper_limit_x
            y = lower_limit_y if side <= 1 else upper_limit_y

            if(x == lower_limit_x and y == lower_limit_y):
                x += directions[side][0]
                y += directions[side][1]

            while (side % 2 == 0 and x  <= upper_limit_x and x  >= lower_limit_x) or (side % 2 == 1 and y  <= upper_limit_y and y  >= lower_limit_y):
                
                if x >= 0 and x < width and y >= 0 and y < height and (x != previous_x or y != previous_y):
                    array.append(grid[x][y])
                    previous_x = x
                    previous_y = y
                x += directions[side][0]
                y += directions[side][1]
    return np.array(array)

def display(maze):
    print("------------------------------------------")
    for i in range(len(maze)):
        line = ""
        for j in range(len(maze[i])):
            if maze[i][j] == 1:
                line += "O"
            else:
                line += "*"
        print(line)
    print("------------------------------------------")

def plotImage(image):
    image = image.reshape(width,height)
    fig = plt.imshow(image)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()

class Search:

    x_scan = open(x_file_name,'r')
    y_scan = open(y_file_name,'r')

    def read(self,scan):
        print("Reading " + scan.name + " for " + str(self.length) + " lines.")
        array = []
        for i in range(self.length):
            layer = []
            for j in range(height*width):
                layer.append(float(scan.read(1)))
            scan.readline()
            array.append(layer)
        return np.array(array).astype('float64').reshape(self.length,width,height)

    def getSolutionPath(self):
        path = []
        v = [0]*self.length
        for i in range(self.length):
            x = start_x
            y = start_y
            array = [(x,y)]
            v[i] = [0]*height
            for j in range(height):
                v[i][j] = [0]*width
            while x != end_x or y != end_y:
                v[i][x][y] = 1
                for j in range(4):
                    move_x = x + directions[j][0]
                    move_y = y + directions[j][1]
                    if v[i][move_x][move_y] == 0 and self.data_y[i][move_x][move_y] == 1:
                        x = move_x
                        y = move_y
                        array.append((x,y))
                        break
            path.append(array)
        return path

    def train(self):
        model.fit(self.spiral, self.next_move, batch_size = batch_size, epochs = 1,verbose = 0)

    def move(self):
        prediction = model.predict(self.spiral)
        for i in range(self.length):
            max_index = 0
            for j in range(1,4):
                if(prediction[i][j] > prediction[i][max_index]):
                    max_index = j
            self.total_moves += 1
            if self.next_move[i][max_index] == 1:
                self.pos[i] = (self.pos[i][0] + directions[max_index][0],self.pos[i][1]+directions[max_index][1])
                if self.pos[i][0] == end_x and self.pos[i][1] == end_y:#exit
                    self.reset()
                else:
                    self.moves[i] += 1
                    self.spiral[i] = self.all_spiral[i][self.moves[i]]
                    self.next_move[i] = self.all_next_move[i][self.moves[i]]
            else:
                self.wrong_moves += 1
    def progress(self,i):
        print(self.name + ": (Epoch",i+1,"/",epoch,")\t | avg incorrect moves",self.wrong_moves/(self.length*self.loops) if self.length > 0 else "-N/A-","move acc: ",self.move_evaluate(),"perfect loops:",self.loop_evaluate(),"looped:",self.loops)
    
    def reset(self):
        self.pos = [(start_x,start_y)]*self.length
        self.previous = [(start_x,start_y)]*self.length
        self.moves = [0]*self.length
        self.loops += 1

        if(self.wrong_moves == 0):
            self.perfect_loops += 1

        self.visited = [0]*self.length
        for i in range(self.length):
            self.visited[i] = [0]*height
            for j in range(height):
                self.visited[i][j] = [0]*width
            self.visited[i][start_x][start_y] = 1
            self.spiral[i] = self.all_spiral[i][self.moves[i]]
            self.next_move[i] = self.all_next_move[i][self.moves[i]]

    def full_reset(self):
        self.wrong_moves = 0
        self.total_moves = 0
        self.loops = 0
        self.perfect_loops = -1
        self.reset()

    def move_evaluate(self):
        return (self.total_moves-self.wrong_moves)/self.total_moves

    def loop_evaluate(self):
        return (self.perfect_loops)/self.loops
        

    def __init__(self,name,length):
        self.name = name
        self.length = length
        self.data_x = self.read(Search.x_scan)
        self.data_y = self.read(Search.y_scan)
        self.spiral = np.zeros((self.length,io_layer))
        self.next_move = np.zeros((self.length,4))

        self.all_spiral = [None]*length
        self.all_next_move = [None]*length
        path = self.getSolutionPath()
        print("Generating all input and output layers")

        for i in range(length):
            self.all_spiral[i] = np.zeros((len(path[i]),io_layer))
            self.all_next_move[i] = np.zeros((len(path[i]),4))
            for j in range(len(path[i])-1):
                self.all_spiral[i][j] = spiralMaze(self.data_x[i],path[i][j][0],path[i][j][1])
                self.all_next_move[i][j] = [0,0,0,0]
                self.all_next_move[i][j][get_direction[1+path[i][j+1][0]-path[i][j][0]][1+path[i][j+1][1]-path[i][j][1]]] = 1
        self.full_reset()

train_search = Search("Training: ",train_count)
test_search = Search("Testing: ",test_count)


for z in range(epoch):
    train_search.full_reset()
    test_search.full_reset()

    run = 0
    while run < run_per_epoch:
        run += 1
        train_search.train()
        train_search.move()
        test_search.move()
    print("---------------------------------------------------------------------------------------------------")
    train_search.progress(z)
    print("---------------------------------------------------------------------------------------------------")
    test_search.progress(z)
    print()
