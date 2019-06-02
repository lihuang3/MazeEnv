import numpy as np, os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d as skel_3d
import random, skimage
ROOT_PATH = os.path.abspath('./MapData')

mapfile = 'map0521'
filename = 'map0521v1'

# Load hand-craft binary maze

mazeData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.txt')).astype(int)
outletData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'a.csv')).astype(int)
np.savetxt('{}/{}.csv'.format(ROOT_PATH, filename), mazeData, fmt= '%3d')
np.savetxt('{}/{}_freespace.csv'.format(ROOT_PATH, filename), mazeData, fmt= '%3d')

dir_dict1 = [[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
dir_dict1_map1 = {0: [-1, 0], 1:[0, -1], 2:[1, 0], 3:[0, 1], 4:[1, 1], 5:[1, -1], 6:[-1, 1], 7:[-1, -1]}
dir_dict1_map2 = {(-1, 0):0, (0, -1):1, (1, 0):2, (0, 1):3, (1, 1):4, (1, -1):5, (-1, 1):6, (-1, -1):7}

dir_dict2 = [[-1, 0], [0, -1], [1, 0], [0, 1]]
dir_dict3 = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

# Object: assign cost-to-go to elements of the centerline
# Method: breadth-first search

import skimage
fig = skimage.io.imread(os.path.join(ROOT_PATH, mapfile+'.png'), as_gray=True)

skel = np.asarray(skel_3d(fig), dtype=int) / 255
# skel[32, 146]  = 1
# skel_Frontier = [[32, 146]]

skel_Frontier = [[109, 480]]

cost = 100
pgradSkel = np.copy(skel)
pgradSkel[skel_Frontier[-1][0], skel_Frontier[-1][1]] = cost


endpoint = []
brchpt = {}
while len(skel_Frontier)>0:
    cost = pgradSkel[skel_Frontier[0][0],skel_Frontier[0][1]]+1
    flag = False
    neighbor = 0
    for dir in dir_dict1:
        new_pt = skel_Frontier[0]+np.array(dir)
        if skel[new_pt[0],new_pt[1]] == 1:
            neighbor += 1
        if pgradSkel[new_pt[0],new_pt[1]] ==1:
            skel_Frontier.append(new_pt)
            pgradSkel[new_pt[0], new_pt[1]] = cost

            flag = True
    if not flag:
        endpoint.append(skel_Frontier[0])
    if neighbor >= 3:
        brchpt[(skel_Frontier[0][0], skel_Frontier[0][1])] = cost - 100
    skel_Frontier.pop(0)

pgradSkel[pgradSkel>=100] -= 99

for item in brchpt:
    skel[item[0], item[1]] = 3


plt.imshow(fig)
plt.show()

brch_len = []

for item in brchpt:
    BSF_frontier = [item]
    cost_init = pgradSkel[item[0], item[1]]

    find = False
    while not find:
        cost = pgradSkel[BSF_frontier[0][0], BSF_frontier[0][1]] -1
        for dir in dir_dict1:
            new_pt = BSF_frontier[0] + np.array(dir)
            if pgradSkel[new_pt[0], new_pt[1]] == cost:
                if cost == 1:
                    brch_len.append(cost_init - cost)
                    find = True
                    break
                try:
                    successor_cost = brchpt[(new_pt[0], new_pt[1])]
                    brch_len.append(cost_init - successor_cost)
                    find = True
                    break
                except KeyError:
                    BSF_frontier.append(new_pt)

        BSF_frontier.pop(0)


brch_len = np.array(brch_len)
brch_len = brch_len[brch_len>1]
print(np.mean(brch_len))