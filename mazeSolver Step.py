from collections import deque
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
#from matplotlib import pyplot as plt

from MazeGen import generate
from MazeSolve import solve


train_file_name = "SuperMaze.txt"
evaluate_file_name = "SolvedSuperMaze.txt"
predict_file_name = "PredictSuperMaze.txt"
model_file_name = "model.yaml"
weight_file_name = "model.h5"

width = 1000
height = 1000
maze_total = 100
recreate = True
directions = [[1,0],[0,1],[-1,0],[0,-1]]

io_layer = width * height - 1
batch_size = 5
epoch = 5
max_run_per_epoch = 100

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

def spiralMaze2(grid,start_x,start_y):
    array = []
    height = len(grid)
    width = len(grid[0])
    distance = max(max(start_x+1,abs(start_x-width)),max(start_y+1,abs(start_y-height)))
    previous = 1000
    for i in range(1,distance):
        upper_limit_x = start_x + i
        upper_limit_y = start_y + i
        lower_limit_x = start_x - i
        lower_limit_y = start_y - i
        side = 0
        if lower_limit_y < 0:
            lower_limit_y = 0
        if upper_limit_y >= height:
            upper_limit_y = height-1
        if lower_limit_x < 0:
            lower_limit_x = 0
        if upper_limit_x >= width:
            upper_limit_x = width-1

        while side < 4:
            x = lower_limit_x if side == 0 or side == 3 else upper_limit_x
            y = lower_limit_y if side <= 1 else upper_limit_y

            if(x == lower_limit_x and y == lower_limit_y):
                x += directions[side][0]
                y += directions[side][1]
            if abs(x - start_x) == i or abs(y - start_y) == i:
                while (abs(x - start_x) == i or abs(y - start_y) == i) and ((side % 2 == 0 and x  <= upper_limit_x and x  >= lower_limit_x) or (side % 2 == 1 and y  <= upper_limit_y and y  >= lower_limit_y)):
                    if previous != grid[x][y]:
                       array.append(grid[x][y])
                       previous = grid[x][y]

                    x += directions[side][0]
                    y += directions[side][1]
            side += 1
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

def read(file_name):
    with open(file_name,'r') as scan:
        firstMaze = []
        for i in range(maze_total):
            maze = []
            for j in range(height*width):
                maze.append(float(scan.read(1)))
            scan.readline()
            firstMaze.append(maze)
    return np.array(firstMaze).astype('float16').reshape(maze_total,width,height)

def plotImage(image):
    image = image.reshape(width,height)
    fig = plt.imshow(image)
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


if (recreate):
    generate(width,height,maze_total,fileName=train_file_name)
    solve(width, height, maze_total, read_file_name = train_file_name, write_file_name = evaluate_file_name)

print("Getting Maze Data")
maze = read(train_file_name)
print("Getting Solution Data")
solution = read(evaluate_file_name)
print("Done Getting Input")

model = Sequential()
model.add(Dense(io_layer, input_shape = (io_layer,)))
model.add(Dense(10,activation = 'relu'))
model.add(Dense(4,activation = 'softmax'))
model.compile(loss = keras.losses.categorical_crossentropy,
    optimizer = Adam(lr = 1e-3),metrics = ['accuracy'])

start_x = 1
start_y = 1

for z in range(epoch):
    pos = [(start_x,start_y)]*maze_total
    previous = [(start_x,start_y)]*maze_total
    is_complete = [False]*maze_total
    did_move = [True]*maze_total
    wrong_moves = 0

    visited = [0]*maze_total
    q = [deque()]*maze_total
    for i in range(maze_total):
        visited[i] = [0]*height
        for j in range(height):
            visited[i][j] = [0]*width
        visited[i][start_x][start_y] = 1
        q[i].append((start_x,start_y))

    spiral = np.zeros((maze_total,io_layer)).astype('float64')
    next_move = np.zeros((maze_total,4)).astype('float64')

    hasRun = True
    run = 0
    while hasRun and run < max_run_per_epoch:
        run += 1
        hasRun = False
        for i in range(maze_total):
            if q[i]:
                if did_move[i]:
                    pos[i] = q[i].popleft()
                    visited[i][pos[i][0]][pos[i][1]] = 1
                    spiral[i] = spiralMaze(maze[i],pos[i][0],pos[i][1])
                    next_move[i] = [0,0,0,0]
                    for j in range(4):
                        attempt = (pos[i][0] + directions[j][0],pos[i][1]+directions[j][1])
                        if solution[i][attempt[0]][attempt[1]] == 1 and visited[i][attempt[0]][attempt[1]] == 0:
                            next_move[i][j] = 1
                            break
                    if pos[i][0] == width-2 and pos[i][1] == height-2:#exit
                        q[i].clear()
                        visited[i] = [0]*height
                        for j in range(height):
                            visited[i][j] = [0]*width
                        visited[i][start_x][start_y] = 1
                        pos[i] = (start_x,start_y)
                        spiral[i] = spiralMaze(maze[i],start_x,start_y)
                        next_move[i] = [0,0,0,0]
                        for j in range(4):
                            attempt = (pos[i][0] + directions[j][0],pos[i][1]+directions[j][1])
                            if solution[i][attempt[0]][attempt[1]] == 1 and visited[i][attempt[0]][attempt[1]] == 0:
                                next_move[i][j] = 1
                                break


        model.fit(spiral, next_move, batch_size = batch_size, epochs = 1,verbose = 0)
        prediction = model.predict(spiral)
        for i in range(maze_total):
            hasRun = True
            did_move[i] = False
            max_index = 0
            for j in range(1,4):
                if(prediction[i][j] > prediction[i][max_index]):
                    max_index = j
            if next_move[i][max_index] == 1:
                did_move[i] = True
                q[i].append((pos[i][0] + directions[max_index][0],pos[i][1]+directions[max_index][1]))
            else:
                wrong_moves += 1
    print("(Epoch",z+1,"/",epoch,")\t | Avg Incorrect Moves",wrong_moves/maze_total)

