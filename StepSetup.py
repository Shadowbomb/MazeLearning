directions = [[1,0],[0,1],[-1,0],[0,-1]]
get_direction = [[0,2],[3,0,1],[0,0]]

fileName = "StepSetUpMaze.txt"


def read(scan):
    array = []
    scan = open(fileName,'r')
    for i in range(self.length):
        layer = []
        for j in range(height*width):
            layer.append(float(scan.read(1)))
        scan.readline()
        array.append(layer)
    return array

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
    return array

def getInputOutputLayers(length,width,height, data_x,data_y,regenerate):
    if regenerate:
        io_layer = width*height-1
        spiral = [0]*length
        next_move = [0]*length
        for i in range(length):
            if i % (length/20) == 0:
                print(str(100.0*i/length) + "%")

            x = start_x
            y = start_y
            path = [(x,y)]
            v = [0]*height
            for j in range(height):
                v[j] = [0]*width
            while x != end_x or y != end_y:
                v[x][y] = 1
                for j in range(4):
                    move_x = x + directions[j][0]
                    move_y = y + directions[j][1]
                    if v[move_x][move_y] == 0 and self.data_y[i][move_x][move_y] == 1:
                        x = move_x
                        y = move_y
                        path.append((x,y))
                        break

        
            spiral[i] = [0]*len(path)
            next_move[i] = [0]*len(path)
            for j in range(len(path[i])-1):
                spiral[i][j] = spiralMaze(self.data_x[i],path[j][0],path[j][1])
                next_move[i][j] = [0,0,0,0]
                next_move[i][j][get_direction[1+path[j+1][0]-path[j][0]][1+path[j+1][1]-path[j][1]]] = 1
    else:


    return spiral,next_move