
from random import randint
from random import shuffle

def isNear(x1,y1,x2,y2):
    return abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1

def display(maze,pos):
    print("------------------------------------------")
    for i in range(len(maze)):
        line = ""
        for j in range(len(maze[i])):
            gotPos = False
            for p in pos:
                if i == p[0] and j == p[1]:
                    line += "#"
                    gotPos = True
                    break
            if(not gotPos):
                if maze[i][j] == 1:
                    line += "O"
                else:
                    line += "*"
        print(line)
    print("------------------------------------------")

def generate(width, height, maze_total,fileName):

    start_x = 1
    start_y = 1

    end_x = width-2
    end_y = height-2


    branch_chance = 2
    
    directions = [[0,-1],[1,0],[0,1],[-1,0]]

    line = ""
    k = 0
    while k < maze_total:
    
        pos = [[start_x,start_y]]
        previous = [[[start_x,start_y]]]
        new = []
	
        maze = [1] * width#Wall = 1 | Path = 0
        for i in range(height):
            maze[i] = [1] * height

        maze[start_x][start_y] = 0

        reachedEndZone = -1
        while len(pos) > 0:
            i = 0
            while i < len(pos):
                #display(maze,pos)
                isFirst = True
                shuffle(directions)
                for dir in directions:
                    x,y = pos[i][0]+dir[0],pos[i][1]+dir[1]
                    if maze[x][y] == 1 and not (x == 0 or x == width-1 or y == 0 or y == height-1):
                        nearWallCount = 0
                        for wall in directions:
                            nearWallCount += maze[x+wall[0]][y+wall[1]]
                        if nearWallCount == 3 and (reachedEndZone == -1 or reachedEndZone == i or not isNear(x,y,end_x,end_y)):
                            if isFirst:
                                isFirst = False
                                previous[i].append([x,y])
                                maze[x][y] = 0
                            elif randint(0,branch_chance) == 0 and not reachedEndZone == i:
                                previous.append([[x,y]])
                                new.append([x,y])
                                maze[x][y] = 0
                        
                pos[i] = previous[i].pop(-1) if isFirst else previous[i][-1]
            
                if isNear(pos[i][0],pos[i][1],end_x,end_y):
                    reachedEndZone = i
                if len(previous[i]) == 0 or pos[i] == previous[i][0]:
                    del pos[i]
                    del previous[i]
                else:
                    i+=1
            while len(new) > 0:
                pos.append(new[0])
                previous.append([new.pop(0)])

        if maze[end_x][end_y] == 0:
            k+=1
            if k % (maze_total/20) == 0:
                print(str(100.0*k/maze_total) + "%")
            for i in range(len(maze)):
                for j in range(len(maze[i])):
                    line += str(maze[i][j])
            if k < maze_total:
                line += "\n"
    
    with open(fileName, 'w') as saveFile:
        saveFile.write(line)
        saveFile.close()

    print("Done")
               



	
