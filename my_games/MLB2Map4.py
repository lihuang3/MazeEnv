import numpy as np, os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d as skel_3d
import random, skimage
ROOT_PATH = os.path.abspath('./MapData')

mapfile = 'map0524'
filename = 'map0524'

# Load hand-craft binary maze

mazeData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.txt')).astype(int)
np.savetxt('{}/{}.csv'.format(ROOT_PATH, mapfile), mazeData, fmt= '%3d')
np.savetxt('{}/{}_freespace.csv'.format(ROOT_PATH, mapfile), mazeData, fmt= '%3d')

dir_dict1 = [[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
dir_dict1_map1 = {0: [-1, 0], 1:[0, -1], 2:[1, 0], 3:[0, 1], 4:[1, 1], 5:[1, -1], 6:[-1, 1], 7:[-1, -1]}
dir_dict1_map2 = {(-1, 0):0, (0, -1):1, (1, 0):2, (0, 1):3, (1, 1):4, (1, -1):5, (-1, 1):6, (-1, -1):7}

dir_dict2 = [[-1, 0], [0, -1], [1, 0], [0, 1]]
dir_dict3 = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

# Object: assign cost-to-go to elements of the centerline
# Method: breadth-first search

# Set goal location
skel = np.asarray(skel_3d(mazeData), dtype=int)

## For Map0522
# for i in range(1,6):
#     skel[31-i, 181+i] = 1
# start = [26, 186]

## For Map 523
# skel[73, 181:186] = 1
# start = [73, 186]

## For map 0524

skel[128, 343:353] = 1
start = [128, 352]

plt.imshow(skel + mazeData)
plt.show()


##++++++++++++++++++++++++++++++++++++++++++++
"""
Configure cost-to-go map
"""
##++++++++++++++++++++++++++++++++++++++++++++

costMap = np.copy(mazeData)
pgrad = np.copy(mazeData)
flowMapCol = 0 * mazeData
flowMapRow = 0 * mazeData
goal = [52, 69]

## for map0522
# v0 [158, 18]
# v1 [22, 56]

## For map0523
# v0 [61, 56]
# v1 [37, 36]

## For map0524
# v0 [52, 69]
# v1 [278, 64]

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
##++++++++++++++++++++++++++++++++++++++++++++



##++++++++++++++++++++++++++++++++++++++++++++
"""
Configure pressure graidient
"""
##++++++++++++++++++++++++++++++++++++++++++++

cost = 100
skel_Frontier = []
BSF_Frontier = []
skel_Frontier.append(start)

# Single source
BSF_Frontier.append(start)
pgrad[BSF_Frontier[-1][0], BSF_Frontier[-1][1]] = cost

# Multi-source
# pgrad[42, 179:184] = 100
# rows, cols = np.where(pgrad == 100)
# for row, col in zip(rows, cols):
#     BSF_Frontier.append([row, col])

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

brch_dict = {}

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
                    brch_dict[item] = new_pt
                    find = True
                    break
                try:
                    brchpt[(new_pt[0], new_pt[1])]
                    if cost_init - cost > 1:
                        brch_level[item] = brch_level[(new_pt[0], new_pt[1])] + 1
                    else:
                        brch_level[item] = brch_level[(new_pt[0], new_pt[1])]
                    brch_dict[item] = new_pt
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

num_robot = 1024

endpoint = outlet_Frontier
loc = np.zeros([num_robot, 2], dtype = np.int32)
robot_cnt = 0
loc_generator = []

endpoint_brchpts_dict = {}

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
                    endpoint_brchpts_dict[(item[0], item[1])] = new_pt
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

for item in endpoint_brchpts_dict:
    brch = endpoint_brchpts_dict[item]
    predecessors = [brch]
    while 1:
        try:
            tmp = brch_dict[(brch[0], brch[1])]
            brch = tmp
            predecessors.append(brch)
        except KeyError:
            break
    endpoint_brchpts_dict[item] = predecessors


##++++++++++++++++++++++++++++++++++++++++++++
"""
Identify brch slopes
"""
##++++++++++++++++++++++++++++++++++++++++++++

brch_slope_upstream = {}
brch_slope_downstream = {}
tail_len = 12
head_len = 12


for item in brchpt:
    BSF_frontier = [item]
    cost_init = pgradSkel[item[0], item[1]]
    tail = []
    head = []
    find = False
    while not find:
        cost = pgradSkel[BSF_frontier[0][0], BSF_frontier[0][1]] -1
        for dir in dir_dict1:
            new_pt = BSF_frontier[0] + np.array(dir)
            if pgradSkel[new_pt[0], new_pt[1]] == cost:
                if cost_init - cost <= tail_len:
                    tail_pt = new_pt
                    tail.append(tail_pt)

                head_pt = new_pt
                head.append(new_pt)

                if cost == 1:
                    find = True
                    break
                try:
                    brchpt[(new_pt[0], new_pt[1])]
                    find = True
                    break
                except KeyError:
                    BSF_frontier.append(new_pt)
        BSF_frontier.pop(0)

    ## Visualize upstream tail
    # tmp_fig = np.copy(skel + mazeData)
    # for i in range(len(tail)):
    #     tmp_fig[tail[i][0], tail[i][1]] = 10
    # plt.imshow(tmp_fig)
    # plt.show()
    slope = item - tail_pt
    brch_slope_upstream[item] = slope
    if cost != 1:
        if len(head) >= head_len:
            head = head[-head_len:]
        slope = head[0] - head[-1]
        brch_slope_downstream[item] = slope
        ## Visualize downstream head
        # tmp_fig = np.copy(skel + mazeData)
        # for i in range(len(head)):
        #     tmp_fig[head[i][0], head[i][1]] = 10
        # plt.imshow(tmp_fig)
        # plt.show()


for item in endpoint:
    BSF_frontier = [item]
    cost_init = pgradSkel[item[0], item[1]]
    head = []
    find = False
    while not find:
        cost = pgradSkel[BSF_frontier[0][0], BSF_frontier[0][1]] -1
        for dir in dir_dict1:
            new_pt = BSF_frontier[0] + np.array(dir)
            if pgradSkel[new_pt[0], new_pt[1]] == cost:
                head_pt = new_pt
                head.append(new_pt)

                if cost == 1:
                    find = True
                    break
                try:
                    brchpt[(new_pt[0], new_pt[1])]
                    find = True
                    break
                except KeyError:
                    BSF_frontier.append(new_pt)
        BSF_frontier.pop(0)

    if len(head) >= head_len:
        head = head[-head_len:]
    slope = head[0] - head[-1]
    brch_slope_downstream[(item[0], item[1])] = slope
    ## Visualize downstream head
    # tmp_fig = np.copy(skel + mazeData)
    # for i in range(len(head)):
    #     tmp_fig[head[i][0], head[i][1]] = 10
    # plt.imshow(tmp_fig)
    # plt.show()


endpt_brch_slope_dict = {}
endpt_brch_control_dict = {}

for idx, item in enumerate(endpoint_brchpts_dict):
    brchs = endpoint_brchpts_dict[item]
    slope = [brch_slope_downstream[item], brch_slope_upstream[(brchs[0][0], brchs[0][1])]]
    endpt_brch_slope_dict[item] = {(brchs[0][0], brchs[0][1]):slope}
    candit1 = np.array([slope[1][1], -slope[1][0]], dtype=np.float)
    candit2 = np.array([-slope[1][1], slope[1][0]], dtype=np.float)
    if np.dot(candit1, slope[0]) > 0:
        control = candit1
    else:
        control = candit2
    dir = dir_dict1[np.argmax(np.dot(np.array(dir_dict1), control))]
    endpt_brch_control_dict[item] = {(brchs[0][0], brchs[0][1]): dir}
    for i in range(1, len(brchs)-1):
        # slope: [downstream_dir_vector, upstream_dir_vector]
        slope = [brch_slope_downstream[(brchs[i-1][0], brchs[i-1][1])], brch_slope_upstream[(brchs[i][0], brchs[i][1])]]
        endpt_brch_slope_dict[item].update({(brchs[i][0], brchs[i][1]): slope})

        candit1 = np.array([slope[1][1], -slope[1][0]], dtype=np.float)
        candit2 = np.array([-slope[1][1], slope[1][0]], dtype=np.float)
        if np.dot(candit1, slope[0]) > 0:
            control = candit1
        else:
            control = candit2
        ## control along perpendicular direction of the upstream
        dir = dir_dict1[np.argmax(np.dot(np.array(dir_dict1), control))]
        ## control along tangent direction of the downstream
        # dir = dir_dict1[np.argmax(np.dot(np.array(dir_dict1), slope[0]))]

        endpt_brch_control_dict[item].update({(brchs[i][0], brchs[i][1]): dir})

        """
        ## Visualize downstream head
        """
        # tmp_fig = np.copy(skel + mazeData)
        # line1 = [brchs[i], brchs[i] + slope[1]]
        # y1, x1 = [line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]]
        # line2 = [brchs[i], brchs[i] + np.dot(dir,5)]
        # y2, x2 = [line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]]
        # plt.imshow(tmp_fig)
        # plt.plot(x1,y1,x2,y2)
        # plt.scatter(brchs[i][1], brchs[i][0])
        # plt.show()

"""
endpt_brch_control_map: [endpt_row, endpt_col, len(brchs), \
            dir1_row, dir1_col, dir2_row, dir2_col, ... ]
"""
endpt_brch_control_map = np.zeros([len(endpoint_brchpts_dict), 100], dtype=np.int16)
endpt_brch_map = np.zeros([len(endpoint_brchpts_dict), 50], dtype=np.int16)

for idx, item in enumerate(endpoint_brchpts_dict):
    brchs = endpoint_brchpts_dict[item]
    endpt_brch_map[idx, :2] = np.array([item[0], item[1]], dtype=np.int16)
    endpt_brch_control_map[idx, :2] = np.array([item[0], item[1]], dtype=np.int16)
    endpt_brch_map[idx,2] = len(brchs) - 1
    endpt_brch_control_map[idx, 2] = len(brchs) - 1
    for i in range(len(brchs)-1):
        endpt_brch_map[idx, 3+2*i:3+2*i+2] = brchs[i]
        endpt_brch_control_map[idx, 3+2*i:3+2*i+2] = endpt_brch_control_dict[item][(brchs[i][0], brchs[i][1])]

np.savetxt('{}/{}_endpt_brch_map.csv'.format(ROOT_PATH, mapfile), endpt_brch_map, fmt = '%5i')
np.savetxt('{}/{}_endpt_brch_control_map.csv'.format(ROOT_PATH, mapfile), endpt_brch_control_map, fmt = '%5i')


# Find patches
detection_map = 0 * mazeData
detection_patch = np.zeros([len(brchpt), 600], dtype=np.int16)
thresh1, thresh2 = 2, 8
for i,brch in enumerate(brchpt):
    tmp_map = 0 * mazeData
    grad = pgrad[brch[0], brch[1]]
    tmp_map[np.logical_and(pgrad<=grad, pgrad>=grad-thresh2)] = 1
    BSF_Frontier = [brch]
    while len(BSF_Frontier)>0:
        for dir in dir_dict1:
            new_pt = BSF_Frontier[0] + np.array(dir)
            if pgrad[new_pt[0], new_pt[1]] <=grad\
                    and pgrad[new_pt[0], new_pt[1]] >=grad-thresh2 \
                    and tmp_map[new_pt[0],new_pt[1]]==1:
                BSF_Frontier.append(new_pt)
                tmp_map[new_pt[0],new_pt[1]] = 2
        BSF_Frontier.pop(0)
    tmp_map[tmp_map<2] = 0
    tmp_map[pgrad>=grad-thresh1] = 0
    # plt.imshow(tmp_map+mazeData)
    # plt.show()
    detection_map += tmp_map
    rows, cols = np.where(tmp_map>0)
    detection_patch[i, :2] = brch
    detection_patch[i, 2] = rows.shape[0]
    tl_row, tl_col = min(rows), min(cols)
    br_row, br_col = max(rows), max(cols)
    detection_patch[i, 3:7] = [tl_row, br_row, tl_col, br_col]
    detection_patch[i, 7:7+rows.shape[0]] = rows
    detection_patch[i, 7+rows.shape[0]: 7+2*rows.shape[0]] = cols

np.savetxt('{}/{}_detect_patch.csv'.format(ROOT_PATH, mapfile), detection_patch, fmt = '%5i')

plt.imshow(detection_map+mazeData)
plt.show()


    # detection_bbox[i,:2] = brch[:]
    # detection_bbox[2:] = [tl_row, tl_col, br_row, br_col]


## Visualize brchpoint predecessors
# for item in endpoint_brchpts_dict:
#     tmp_fig = np.copy(skel + mazeData)
#     brchs = endpoint_brchpts_dict[item]
#     for i in range(len(brchs) ):
#         tmp_fig[brchs[i][0], brchs[i][1]] = 10
#     plt.imshow(tmp_fig)
#     plt.show()

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
        if pgrad[new_pt[0], new_pt[1]] == cost - 1 or pgrad[new_pt[0], new_pt[1]] == cost:
            candit.append(dir)
    dirs = candit
    prob = np.ones([len(dirs)]) / len(dirs)
    flow_dict[(row, col)] = [dirs, prob]
##++++++++++++++++++++++++++++++++++++++++++++



def render(loc):
    plt.gcf().clear()
    robot_marker = 150
    render_image = np.copy(0 * mazeData).astype(np.int16)
    for i in range(len(loc)):
        render_image[loc[i, 0] - 1:loc[i, 0] + 2,
        loc[i, 1] - 1:loc[i, 1] + 2] += robot_marker

    row, col = np.nonzero(render_image)
    min_robots = 150.
    max_robots = float(np.max(render_image))
    # rgb_render_image = np.stack((render_image+self.maze*255,)*3, -1)
    rgb_render_image = np.stack(
        (render_image + mazeData * 128, render_image + mazeData * 228, render_image + mazeData * 255), -1)
    rgb_render_image[rgb_render_image[:, :, :] == 0] = 255

    for i in range(row.shape[0]):
        value = render_image[row[i], col[i]]
        ratio = 0.4 + 0.5 * max(value - min_robots, 0) / (max_robots - min_robots)
        ratio = min(0.9, max(0.4, ratio))
        b = 180
        g = 180 * (1 - ratio)
        r = 180 * (1 - ratio)

        for j, rgb in enumerate([r, g, b]):
            rgb_render_image[row[i], col[i], j] = np.uint8(rgb)

    plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)
    plt.show(False)
    plt.pause(0.0001)

for brch in brch_level:
    if brch_level[brch] == 1:
        first_brch = brch
        break
loc_int = np.copy(loc)
flow_map = {}
visit_stats = 0 * mazeData

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
    visit_stats[loc[:,0], loc[:,1]] += 1
    # render(loc)

    return stay == len(loc)


round = 0
diff = len(np.where(pgrad>1)[0])-len(flow_map)
while round<50:
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

np.savetxt('{}/{}_flowstats.csv'.format(ROOT_PATH, mapfile), output, fmt = '%.1f')
np.savetxt('{}/{}_pgrad.csv'.format(ROOT_PATH, mapfile), pgrad, fmt = '%5i')
np.savetxt('{}/{}_visit.csv'.format(ROOT_PATH, mapfile), visit_stats, fmt = '%5i')


