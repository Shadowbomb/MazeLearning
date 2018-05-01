from collections import deque

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

write_line = ""
directions = [[0,-1],[1,0],[0,1],[-1,0]]
read_file_name = "SuperMaze.txt"
write_file_name = "SolvedSuperMaze.txt"

with open(read_file_name,'r') as scan:
    maze_count = 1000
    width = 64
    height = 64

    for i in range(maze_count):
        if maze_count >= 4 and i % (maze_count/4) == 0:
            print(str(100.0*i/maze_count) + "%")
        maze = []
        for j in range(height):
            line = []
            for k in range(width):
                line.append(int(scan.read(1)))
            maze.append(line)
        scan.readline()
        q = deque()
        pos = (1,1)
        maze[pos[0]][pos[1]] = 1

        previous = [None] * width
        for j in range(height):
            previous[j] = [None] * height

        path = [0] * width
        for j in range(height):
            path[j] = [0] * height
            
        q.append(pos)
        while q:

            pos = q.popleft()
            if pos[0] == width-2 and pos[1] == height-2:
                    while not pos == None:
                        path[pos[0]][pos[1]] = 1
                        pos = previous[pos[0]][pos[1]]
                    break
            maze[pos[0]][pos[1]] = 1
            for dir in directions:
                new = (pos[0] + dir[0],pos[1]+dir[1])
                if maze[new[0]][new[1]] == 0:
                    q.append(new)
                    previous[new[0]][new[1]] = pos
        for j in range(len(path)):
            for k in range(len(path[j])):
                write_line += str(path[j][k])
        if i < maze_count - 1:
            write_line += "\n"


    scan.close()
with open(write_file_name, 'w') as saveFile:
    saveFile.write(write_line)
    saveFile.close()
print("100.0%\nDone")
