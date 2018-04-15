import random

width = 100
height = 100
difficulty = 5

walls = []
maze = [0] * width

for i in range(height):
    maze[i] = [0] * height

maze[0][1] = 1

walls.append([1, 1, 1])

def addWalls():
    walls.append([X + 1, Y, 1])
    walls.append([X - 1, Y, 2])
    walls.append([X, Y + 1, 3])
    walls.append([X, Y - 1, 4])
    maze[X][Y] = 1

while len(walls) > 0:
    if len(walls) > difficulty:
        current = len(walls) - round(random.random() * difficulty) - 1
    else:
        current = round(random.random() * (len(walls) - 1))

    cell = walls[current]
    walls.pop(current)

    X = cell[0]
    Y = cell[1]
    Dir = cell[2]

    if Dir == 1:
        if X + 1 < width and maze[X][Y] == 0 and maze[X + 1][Y] == 0 and maze[X + 1][Y - 1] == 0 and maze[X + 1][Y + 1] == 0 and maze[X][Y - 1] == 0 and maze[X][Y + 1] == 0:
            addWalls()
    elif Dir == 2:
        if X - 1 >  - 1 and maze[X][Y] == 0 and maze[X - 1][Y] == 0 and maze[X - 1][Y - 1] == 0 and maze[X - 1][Y + 1] == 0 and maze[X][Y - 1] == 0 and maze[X][Y + 1] == 0:
            addWalls()
    elif Dir == 3:
        if Y + 1 < height and maze[X][Y] == 0 and maze[X][Y + 1] == 0 and maze[X - 1][Y + 1] == 0 and maze[X + 1][Y + 1] == 0 and maze[X - 1][Y] == 0 and maze[X + 1][Y] == 0:
            addWalls()
    elif Dir == 4:
        if Y - 1 >  - 1 and maze[X][Y] == 0 and maze[X][Y - 1] == 0 and maze[X - 1][Y - 1] == 0 and maze[X + 1][Y - 1] == 0 and maze[X - 1][Y] == 0 and maze[X + 1][Y] == 0:
            addWalls()

for y in range(0, height):
    line = ''
    for x in range(0, width):
        if (maze[x][y] == 0):
            line += '0'
        else:
            line += '1'
    print(line)