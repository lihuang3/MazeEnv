import numpy as np, os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d as skel_3d
import random, skimage
ROOT_PATH = os.path.abspath('./MapData')

mapfile = 'map0522'
filename = 'map0522v1'

# Load hand-craft binary maze

mazeData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.txt')).astype(int)
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
skel[42, 178:185] = 1
start = [42, 184]
#

plt.imshow(skel + mazeData)
plt.show()

# Initialize centerline cost-to-go map
costMap = np.copy(mazeData)
pgrad = np.copy(mazeData)
flowMapCol = 0 * mazeData
flowMapRow = 0 * mazeData
goal = [22, 56]
#[158, 18]
#[22, 56]


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

skel_Frontier.append(start)

# BSF_Frontier.append(start)
# pgrad[BSF_Frontier[-1][0], BSF_Frontier[-1][1]] = cost

# Manual step
pgrad[35:50, 184] = 100
rows, cols = np.where(pgrad == 100)

for row, col in zip(rows, cols):
    BSF_Frontier.append([row, col])

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



##++++++++++++++++++++++++++++++++++++++++++++
"""
Get branch points and hierachies
"""
##++++++++++++++++++++++++++++++++++++++++++++

brchpt = {}
endpoint = []
while len(skel_Frontier)>0:
    cost = pgradSkel[skel_Frontier[0][0],skel_Frontier[0][1]]+1
    flag = False
    neighbor = 0
    for dir in dir_dict1:
        new_pt = skel_Frontier[0]+np.array(dir)
        if skel[new_pt[0], new_pt[1]] == 1:
            neighbor += 1
        if pgradSkel[new_pt[0],new_pt[1]] ==1:
            skel_Frontier.append(new_pt)
            pgradSkel[new_pt[0], new_pt[1]] = cost
            flag = True
    if not flag:
        endpoint.append(skel_Frontier[0])
    if neighbor >= 3:
        isbrch = True
        for dir in dir_dict1:
            try:
                brchpt[(skel_Frontier[0][0]+dir[0], skel_Frontier[0][1]+dir[1])]
                isbrch = False
            except KeyError:
                pass
        if isbrch:
            brchpt[(skel_Frontier[0][0], skel_Frontier[0][1])] = cost - 100
    skel_Frontier.pop(0)

outlet_Frontier = endpoint
pgradSkel[pgradSkel>=100] -= 99
endpoint = np.asarray(endpoint, dtype=np.int16)

tmp_fig = np.copy(pgradSkel)
tmp_fig[endpoint[:,0], endpoint[:,1]] = 500
plt.imshow(tmp_fig)
plt.show()


brch_level = {}

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
                    brch_level[item] = 1
                    find = True
                    break
                try:
                    brchpt[(new_pt[0], new_pt[1])]
                    if cost_init - cost > 1:
                        brch_level[item] = brch_level[(new_pt[0], new_pt[1])] + 1
                    else:
                        brch_level[item] = brch_level[(new_pt[0], new_pt[1])]
                    find = True
                    break
                except KeyError:
                    BSF_frontier.append(new_pt)

        BSF_frontier.pop(0)


tmp_fig = np.copy(mazeData)
for item in brch_level:
    tmp_fig[item[0],item[1]] += 10*brch_level[item]

plt.imshow(tmp_fig)
plt.show()
##++++++++++++++++++++++++++++++++++++++++++++



##++++++++++++++++++++++++++++++++++++++++++++
"""
Distribute robots in endpoints
"""
##++++++++++++++++++++++++++++++++++++++++++++

thresh = 10
num_robot = 256 + 128

endpoint = outlet_Frontier
loc = np.zeros([num_robot, 2], dtype = np.int32)
robot_cnt = 0
loc_generator = []

for item in endpoint:
    BSF_Frontier = []
    BSF_Frontier.append(item)

    find = False
    while not find:
        cost = pgradSkel[BSF_Frontier[0][0], BSF_Frontier[0][1]] -1
        for dir in dir_dict1:
            new_pt = BSF_Frontier[0] + np.array(dir)
            if pgradSkel[new_pt[0], new_pt[1]] == cost:
                try:
                    level = brch_level[(new_pt[0], new_pt[1])]
                    find = True
                    break
                except KeyError:
                    BSF_Frontier.append(new_pt)

        BSF_Frontier.pop(0)


    cost = 100
    pgradOutlet = np.copy(mazeData)
    pgradOutlet[item[0],item[1]] = cost

    BSF_Frontier = [item]
    local_robot = 1
    while len(BSF_Frontier)>0 and local_robot < 1.5*num_robot/(2**level):
        cost = pgradOutlet[BSF_Frontier[0][0],BSF_Frontier[0][1]]+1
        for dir in dir_dict2:
            new_pt = BSF_Frontier[0]+np.array(dir)
            if pgradOutlet[new_pt[0],new_pt[1]] == 1.0:
                local_robot += 1
                BSF_Frontier.append(new_pt)
                pgradOutlet[new_pt[0], new_pt[1]] = cost
        BSF_Frontier.pop(0)

    rows, cols = np.where(pgradOutlet>=100)
    idx = np.random.choice(len(rows), num_robot//(2**level))
    loc[robot_cnt:robot_cnt + num_robot//(2**level), 0] = rows[idx]
    loc[robot_cnt:robot_cnt + num_robot//(2**level), 1] = cols[idx]
    robot_cnt += num_robot//(2**level)
    loc_generator.append([num_robot//(2**level), rows, cols])
tmp_fig = np.copy(skel+mazeData)
for i in range(num_robot):
    tmp_fig[loc[i,0], loc[i, 1]] = 10
plt.imshow(tmp_fig)
plt.show()
##++++++++++++++++++++++++++++++++++++++++++++


##++++++++++++++++++++++++++++++++++++++++++++
"""
Assign prob to each direction at each location
"""
##++++++++++++++++++++++++++++++++++++++++++++

rows, cols = np.where(pgrad>1)
flow_dict = {}

noSuccessor = []
for row, col in zip(rows, cols):
    done = False
    cost = pgrad[row, col]
    candit = []
    for dir in dir_dict1:
        new_pt = np.array([row, col]) + np.array(dir)
        if pgrad[new_pt[0], new_pt[1]] == cost - 1:
            candit.append(dir)
    dirs = candit
    prob = np.ones([len(dirs)]) / len(dirs)
    flow_dict[(row, col)] = [dirs, prob]
##++++++++++++++++++++++++++++++++++++++++++++

for brch in brch_level:
    if brch_level[brch] == 1:
        first_brch = brch
        break
loc_int = np.copy(loc)
flow_map = {}

def step():
    stay = 0
    for i in range(len(loc)):
        row, col = loc[i,:]

        if pgrad[row, col] == 1:
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
while round<80:
    print(round, diff)
    # loc = np.copy(loc_int)
    robot_cnt = 0
    for item in loc_generator:
        local_robot, rows, cols = item[0], item[1], item[2]
        idx = np.random.choice(len(rows), local_robot)
        loc[robot_cnt:robot_cnt + local_robot, 0] = rows[idx]
        loc[robot_cnt:robot_cnt + local_robot, 1] = cols[idx]
        robot_cnt += local_robot



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
np.savetxt('{}/{}_pgrad.csv'.format(ROOT_PATH, filename), pgrad, fmt = '%5i')

