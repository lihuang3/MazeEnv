import numpy as np, os
import matplotlib.pyplot as plt

ROOT_PATH = os.path.abspath('./MapData')

mapfile = 'map1204'
filename = 'map1204v1'
action_space = 4
# Load hand-craft binary maze
mazeData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.txt')).astype(int)


np.savetxt('{}/{}.csv'.format(ROOT_PATH, filename), mazeData, fmt= '%3d')
np.savetxt('{}/{}_freespace.csv'.format(ROOT_PATH, filename), mazeData, fmt= '%3d')


# Object: assign cost-to-go to elements of the centerline
# Method: breadth-first search
BSF_Frontier = []

# Set goal location
goal = np.array([103, 137])

# Initialize centerline cost-to-go map
costMap = np.copy(mazeData)
BSF_Frontier.append(goal)
cost = 100
costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]] = cost

if action_space ==4:
    while len(BSF_Frontier)>0:
        cost = costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]]+1
        for i in range(4):
            new_pt = BSF_Frontier[0]+np.array([np.cos(i*np.pi/2),np.sin(i*np.pi/2)]).astype(int)
            if costMap[new_pt[0],new_pt[1]] == 1.0:
                BSF_Frontier.append(new_pt)
                costMap[new_pt[0], new_pt[1]] = cost
        BSF_Frontier.pop(0)
elif action_space==8:
    while len(BSF_Frontier) > 0:
        cost = costMap[BSF_Frontier[0][0], BSF_Frontier[0][1]] + 1
        for i in range(-1,2):
            for j in range(-1,2):
                if not (i==0 and j==0):
                    new_pt = BSF_Frontier[0] + np.array([i,j]).astype(int)
                    if costMap[new_pt[0], new_pt[1]] == 1.0:
                        BSF_Frontier.append(new_pt)
                        costMap[new_pt[0], new_pt[1]] = cost
        BSF_Frontier.pop(0)
costMap -= 99*mazeData


np.savetxt('{}/{}_costmap.csv'.format(ROOT_PATH, filename), costMap, fmt = '%3d')



