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

# Set goal location
skel = np.asarray(skel_3d(mazeData), dtype=int)
skel[32, 146]  = 1

# plt.imshow(skel)
# plt.show()
rows, cols = np.where(np.logical_and( outletData == 9, skel == 1))

# Initialize centerline cost-to-go map
costMap = np.copy(mazeData)
pgrad = np.copy(mazeData)
flowMapCol = 0 * mazeData
flowMapRow = 0 * mazeData
goal = [19, 44]

# [19, 44]

# [104, 49]




BSF_Frontier = []
BSF_Frontier.append(goal)
cost = 100
costMap[goal[0],goal[1]] = cost
while len(BSF_Frontier)>0:
    cost = costMap[BSF_Frontier[0][0],BSF_Frontier[0][1]] + 1
    for dir in dir_dict2:
        new_pt = BSF_Frontier[0]+np.array(dir)
        if costMap[new_pt[0],new_pt[1]] == 1:
            BSF_Frontier.append(new_pt)
            costMap[new_pt[0], new_pt[1]] = cost

    BSF_Frontier.pop(0)

costMap[costMap>=100] -= 99
np.savetxt('{}/{}_costmap.csv'.format(ROOT_PATH, filename), costMap, fmt = '%5i')



cost = 100
skel_Frontier = []
BSF_Frontier = []

for row, col in zip(rows, cols):
    BSF_Frontier.append([row, col])
    skel_Frontier.append([row, col])
    pgrad[BSF_Frontier[-1][0], BSF_Frontier[-1][1]] = cost



while len(BSF_Frontier)>0:
    cost = pgrad[BSF_Frontier[0][0],BSF_Frontier[0][1]] + 1
    flag = False
    for dir in dir_dict1:
        new_pt = BSF_Frontier[0]+np.array(dir)
        if skel[new_pt[0],new_pt[1]] == 1.0 and pgrad[new_pt[0],new_pt[1]] == 1:
            BSF_Frontier.append(new_pt)
            pgrad[new_pt[0], new_pt[1]] = cost
    for dir in dir_dict1:
        new_pt = BSF_Frontier[0]+np.array(dir)
        if skel[new_pt[0],new_pt[1]] != 1.0 and pgrad[new_pt[0],new_pt[1]] == 1:
            BSF_Frontier.append(new_pt)
            pgrad[new_pt[0], new_pt[1]] = cost

    BSF_Frontier.pop(0)

pgrad[pgrad>=100] -= 99

cost = 100
pgradSkel = np.copy(skel)
pgradSkel[skel_Frontier[-1][0], skel_Frontier[-1][1]] = cost


endpoint = []
while len(skel_Frontier)>0:
    cost = pgradSkel[skel_Frontier[0][0],skel_Frontier[0][1]]+1
    flag = False
    for dir in dir_dict1:
        new_pt = skel_Frontier[0]+np.array(dir)
        if pgradSkel[new_pt[0],new_pt[1]] ==1:
            skel_Frontier.append(new_pt)
            pgradSkel[new_pt[0], new_pt[1]] = cost
            flag = True
    if not flag:
        endpoint.append(skel_Frontier[0])
    skel_Frontier.pop(0)

outlet_Frontier = endpoint
endpoint = np.asarray(endpoint, dtype=np.int16)
pgradSkel[endpoint[:,0], endpoint[:,1]] = 1000

plt.imshow(pgradSkel)
plt.show()

thresh = 7
cost = 100
pgradOutlet = np.copy(mazeData)
for item in outlet_Frontier:
    pgradOutlet[item[0],item[1]] = cost

while len(outlet_Frontier)>0 and (cost<=thresh+100):
    cost = pgradOutlet[outlet_Frontier[0][0],outlet_Frontier[0][1]]+1
    for dir in dir_dict2:
        new_pt = outlet_Frontier[0]+np.array(dir)
        if pgradOutlet[new_pt[0],new_pt[1]] == 1.0:
            outlet_Frontier.append(new_pt)
            pgradOutlet[new_pt[0], new_pt[1]] = cost
    outlet_Frontier.pop(0)
pgradOutlet[pgradOutlet>=100] -= 90
# plt.imshow(pgradOutlet)
# plt.show()



p1, p2 = 0.6, 0.4
rows, cols = np.where(pgrad>1)
flow_dict = {}

noSuccessor = []
for row, col in zip(rows, cols):
    done = False
    cost = pgrad[row, col]
    candit = []
    canditSkel = []
    p1, p2 = 0.8, 0.2

    for dir in dir_dict1:
        new_pt = np.array([row, col]) + np.array(dir)
        if pgrad[new_pt[0], new_pt[1]] == cost - 1:
            if skel[ new_pt[0], new_pt[1] ]  == 1:
                canditSkel.append(dir)
            else:
                candit.append(dir)

    if p2>0:
        a, b = len(candit), len(canditSkel)
        if a>0 and b>0:
            candit.extend(canditSkel)
            dirs = candit
            prob = np.ones([len(dirs)])
            prob[:a] = p1/a
            prob[a:] = p2/b
        else:
            dirs = candit if a>0 else canditSkel
            prob = np.ones([len(dirs)]) * 1.0 / len(dirs)
    else:
        candit.extend(canditSkel)
        dirs = candit
        prob = np.ones([len(dirs)]) * 1.0 / len(dirs)

    flow_dict[(row, col)] = [dirs, prob]


rows, cols = np.where(pgradOutlet >= 10)

num = len(rows)
loc = np.zeros([num, 2], dtype = np.int32)
loc[:,0] = rows
loc[:,1] = cols
loc_int = np.copy(loc)
flow_map = {}


def step():
    stay = 0
    for i in range(len(loc)):
        row, col = loc[i,:]
        if pgrad[row, col] ==1:
            dir = np.array([0,0])
            stay += 1
        else:
            item = flow_dict[(row, col)]
            dirs, probs = item[0], item[1]
            idx = np.random.choice(len(dirs), 1, p=probs)[0]
            dir = dirs[idx]
            loc[i,:] += np.array(dir)
            row, col = loc[i,:]
            try:
                flow_map[(row, col)]
                flow_map[(row, col)][dir_dict1_map2[(dir[0],dir[1])]] += 1
            except KeyError:
                flow_map[(row, col)] = np.zeros([8], dtype=np.int32)
                flow_map[(row, col)][dir_dict1_map2[(dir[0],dir[1])]] = 1
    return stay == len(loc)


round = 0
diff = len(np.where(pgrad>1)[0])-len(flow_map)
while round<50:
    print(round, diff)
    loc = np.copy(loc_int)
    while 1:
        done = step()
        if done:
            break
    round += 1
    diff = len(np.where(pgrad>1)[0])-len(flow_map)

for i, item in enumerate(flow_map):
    mazeData[item[0],item[1]] = np.sum(flow_map[item])
# plt.imshow(mazeData)
# plt.show()

h, w = np.shape(mazeData)
output = np.zeros([h*w, 9], dtype = np.float32)
for i, item in enumerate(flow_map):
    output[item[0]*w + item[1], 1:] = flow_map[item].astype(np.float32)
validout = np.sum(output, axis=1)
idx = np.where(validout>=10)[0]
output[idx,0] = np.array(idx).astype(np.float32)
output = output[idx,:]

np.savetxt('{}/{}_flowstats.csv'.format(ROOT_PATH, filename), output, fmt = '%.1f')
np.savetxt('{}/{}_pgrad.csv'.format(ROOT_PATH, filename), prad, fmt = '%5i')

