import numpy as np, os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d as skel_3d
import random, skimage
ROOT_PATH = os.path.abspath('./MapData')

mapfile = 'map0521'
filename = 'map0521'

# Load hand-craft binary maze

mazeData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.txt')).astype(int)
outletData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'a.csv')).astype(int)
np.savetxt('{}/{}.csv'.format(ROOT_PATH, filename), mazeData, fmt= '%3d')
np.savetxt('{}/{}_freespace.csv'.format(ROOT_PATH, filename), mazeData, fmt= '%3d')

# Object: assign cost-to-go to elements of the centerline
# Method: breadth-first search
BSF_Frontier = []

# Set goal location
skel = np.asarray(skel_3d(mazeData), dtype=int)
skel[32, 146]  = 1

# plt.imshow(skel)
# plt.show()
rows, cols = np.where(np.logical_and( outletData == 9, skel == 1))

# Initialize centerline cost-to-go map
costMap = np.copy(mazeData)
flowMapCol = 0 * mazeData
flowMapRow = 0 * mazeData

cost = 100
skel_Frontier = []
costMapSkel = np.copy(skel)

for row, col in zip(rows, cols):
    BSF_Frontier.append([row, col])
    costMap[BSF_Frontier[-1][0], BSF_Frontier[-1][1]] = cost
    costMapSkel[BSF_Frontier[-1][0], BSF_Frontier[-1][1]] = cost

dir_dict1 = [[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
dir_dict2 = [[-1, 0], [0, -1], [1, 0], [0, 1]]
dir_dict3 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
skel_Frontier = []

while len(BSF_Frontier)>0:
    cost = costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]]+1
    skel_Frontier.append(BSF_Frontier[0])
    for dir in dir_dict1:
        new_pt = BSF_Frontier[0]+np.array(dir)
        if skel[new_pt[0],new_pt[1]] == 1.0 and costMap[new_pt[0],new_pt[1]] == 1:
            BSF_Frontier.append(new_pt)
            costMap[new_pt[0], new_pt[1]] = cost
    for dir in dir_dict1:
        new_pt = BSF_Frontier[0]+np.array(dir)
        if skel[new_pt[0],new_pt[1]] != 1.0 and costMap[new_pt[0],new_pt[1]] == 1:
            BSF_Frontier.append(new_pt)
            costMap[new_pt[0], new_pt[1]] = cost

    BSF_Frontier.pop(0)

costMap[costMap>=100] -= 99



for item in skel_Frontier:
    costMap[item[0],item[1]] = costMap[item[0],item[1]]*10 + 500


outlet = []
rows, cols = np.where(costMap>0)

for row, col in zip(rows, cols):
    done = False
    cost = costMap[row, col]
    candit = []
    candit0 = []
    for dir in dir_dict1:
        new_pt = np.array([row, col]) + np.array(dir)
        if costMap[new_pt[0],new_pt[1]] == cost + 1:
            candit.append(dir)
        elif costMap[new_pt[0],new_pt[1]] == cost:
            candit0.append(dir)
    if len(candit) > 0:
        dir = candit[ random.sample(range(len(candit)), 1)[0] ]
        flowMapCol[row, col], flowMapRow[row, col] = dir[0], dir[1]
        done = True
    elif len(candit0)>0:
        dir = candit0[ random.sample(range(len(candit0)), 1)[0] ]
        flowMapCol[row, col], flowMapRow[row, col] = dir[0], dir[1]
        done = True




#
# np.savetxt('{}/{}_costmap.csv'.format(ROOT_PATH, filename), costMap, fmt = '%5i')
# np.savetxt('{}/{}_flowmapRow.csv'.format(ROOT_PATH, filename), flowMapRow, fmt = '%2i')
# np.savetxt('{}/{}_flowmapCol.csv'.format(ROOT_PATH, filename), flowMapCol, fmt = '%2i')
#
#
#
