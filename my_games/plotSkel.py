import numpy as np, os
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize_3d as skel_3d
import random, skimage
ROOT_PATH = os.path.abspath('./MapData')

mapfile = 'map0523'
filename = 'map0523'

# Load hand-craft binary maze
raw_img = plt.imread(os.path.join(ROOT_PATH, mapfile+'.png'))
mazeData = np.loadtxt(os.path.join(ROOT_PATH, mapfile+'.txt')).astype(int)
h, w = np.shape(raw_img)[:2]
h1, w1 = mazeData.shape

dir_dict1 = [[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
dir_dict1_map1 = {0: [-1, 0], 1:[0, -1], 2:[1, 0], 3:[0, 1], 4:[1, 1], 5:[1, -1], 6:[-1, 1], 7:[-1, -1]}
dir_dict1_map2 = {(-1, 0):0, (0, -1):1, (1, 0):2, (0, 1):3, (1, 1):4, (1, -1):5, (-1, 1):6, (-1, -1):7}

dir_dict2 = [[-1, 0], [0, -1], [1, 0], [0, 1]]
dir_dict3 = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

# Object: assign cost-to-go to elements of the centerline
# Method: breadth-first search

# Set goal location
bw = raw_img[:,:,0]
bw[bw<1] = 0
skel = np.asarray(skel_3d(bw), dtype=np.int16)/255
tmp_fig = np.copy(raw_img)
tmp_fig[skel==1,:] = 0

## For Map0522
# start = [77, 453]

## For Map 523
start = [183, 450]

## For map 0524
# start = [213, 570]

plt.imshow(tmp_fig)
plt.show()


##++++++++++++++++++++++++++++++++++++++++++++
"""
Configure cost-to-go map
"""
##++++++++++++++++++++++++++++++++++++++++++++

pgrad = np.copy(bw)
flowMapCol = 0 * bw
flowMapRow = 0 * bw

goal1 =[61, 56]
goal2 = [37, 36]
goal1 = [int(goal1[0]*h/h1), int(goal1[1]*w/w1)]
goal2 = [int(goal2[0]*h/h1), int(goal2[1]*w/w1)]
## for map0522
# v0 [158, 18]
# v1 [22, 56]

## For map0523
# v0 [61, 56]
# v1 [37, 36]

## For map0524
# v0 [52, 69]
# v1 [278, 64]


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


tmp_fig = np.copy(pgrad)/np.max(pgrad)


import matplotlib as mpl
fig = plt.gcf()
fig.set_size_inches(10, 10)
rows, cols = np.where(skel>0)
for row, col in zip(rows, cols):
    tmp_fig[row, col] = 0
plt.imshow(tmp_fig, cmap=mpl.cm.get_cmap("jet"))


circle1 = plt.Circle((goal1[1], goal1[0]), 15, linestyle='-', color='blue',
                    linewidth=2, fill=False)
circle2 = plt.Circle((goal2[1], goal2[0]), 15, linestyle='-', color='green',
                    linewidth=2, fill=False)

plt.gcf().gca().add_artist(circle1)
plt.gcf().gca().add_artist(circle2)


for item in brch_level:
    circle = plt.Circle((item[1], item[0]), 4/brch_level[item] , linestyle='-', color='black',
                        linewidth=2, fill=True)
    plt.gcf().gca().add_artist(circle)


plt.colorbar(fraction=0.08, pad=0.04)


plt.axis('off')
fig.tight_layout()
fig.subplots_adjust \
    (top=0.979,
bottom=0.024,
left=0.051,
right=0.979,
hspace=0.2,
wspace=0.2)
# plt.savefig( os.path.join(ROOT_PATH, filename+'_profile.png'), pad_inches=0.0, dpi=100)


plt.show()
##++++++++++++++++++++++++++++++++++++++++++++


##++++++++++++++++++++++++++++++++++++++++++++
"""
Distribute robots in endpoints
"""
##++++++++++++++++++++++++++++++++++++++++++++

num_robot = 512

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
    pgradOutlet = np.copy(bw)
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

    # # Visualize upstream tail
    # tmp_fig = np.copy(skel + bw)
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
        # # Visualize downstream head
        # tmp_fig = np.copy(skel + bw)
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
    # tmp_fig = np.copy(skel + bw)
    # for i in range(len(head)):
    #     tmp_fig[head[i][0], head[i][1]] = 10
    # plt.imshow(tmp_fig)
    # plt.show()


endpt_brch_slope_dict = {}
endpt_brch_control_dict = {}

arrow1 = set()
arrow2 = set()
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

    """
    Arrow visual
    """
    # --------------------------
    line1 = [brchs[0] - 2 * slope[1], brchs[0]]
    y1, x1 = (line1[0][0], line1[1][0]), (line1[0][1], line1[1][1])
    line2 = [brchs[0] + np.dot(dir, 8), brchs[0] + np.dot(dir, 18)]
    y2, x2 = (line2[0][0], line2[1][0]), (line2[0][1], line2[1][1])
    arrow1.add((x1, y1))
    arrow2.add((x2, y2))
    # -----------------------------

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
        # tmp_fig = np.copy(skel + bw)
        # line1 = [brchs[i] - slope[1], brchs[i]]
        # y1, x1 = [line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]]
        # line2 = [brchs[i], brchs[i] + np.dot(dir,5)]
        # y2, x2 = [line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]]
        # plt.imshow(tmp_fig)
        # plt.plot(x1,y1,x2,y2)
        # plt.scatter(brchs[i][1], brchs[i][0])
        # plt.show()

        """
        Arrow visual
        """
        # --------------------------
        line1 = [brchs[i] - 2*slope[1], brchs[i]]
        y1, x1 = (line1[0][0], line1[1][0]), (line1[0][1], line1[1][1])
        line2 = [brchs[i] + np.dot(dir,5), brchs[i] + np.dot(dir,15)]
        y2, x2 = (line2[0][0], line2[1][0]), (line2[0][1], line2[1][1])
        arrow1.add((x1,y1))
        arrow2.add((x2,y2))
        # --------------------------

fig = plt.gcf()
fig.set_size_inches(10, 10)

tmp_fig = np.copy(raw_img)
plt.imshow(tmp_fig)
for item in arrow1:
    x, y = item[0], item[1]
    plt.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], head_width=8, head_length=8, fc='r', ec='r', width=2)
for item in arrow2:
    x, y = item[0], item[1]
    plt.arrow(x[0], y[0], x[1]-x[0], y[1]-y[0], head_width=5, head_length=5, fc='b', ec='b', width=1)

plt.axis('off')
fig.tight_layout()
fig.subplots_adjust \
    (top=0.98,
bottom=0.04,
left=0.04,
right=0.985,
hspace=0.2,
wspace=0.2)
plt.savefig( os.path.join(ROOT_PATH, filename+'_arrow.png'), pad_inches=0.0, dpi=100)

plt.show()

"""
endpt_brch_control_map: [endpt_row, endpt_col, len(brchs), \
            dir1_row, dir1_col, dir2_row, dir2_col, ... ]
"""
endpt_brch_control_map = np.zeros([len(endpoint_brchpts_dict), 500], dtype=np.int16)
endpt_brch_map = np.zeros([len(endpoint_brchpts_dict), 250], dtype=np.int16)

for idx, item in enumerate(endpoint_brchpts_dict):
    brchs = endpoint_brchpts_dict[item]
    endpt_brch_map[idx, :2] = np.array([item[0], item[1]], dtype=np.int16)
    endpt_brch_control_map[idx, :2] = np.array([item[0], item[1]], dtype=np.int16)
    endpt_brch_map[idx,2] = len(brchs) - 1
    endpt_brch_control_map[idx, 2] = len(brchs) - 1
    for i in range(len(brchs)-1):
        endpt_brch_map[idx, 3+2*i:3+2*i+2] = brchs[i]
        endpt_brch_control_map[idx, 3+2*i:3+2*i+2] = endpt_brch_control_dict[item][(brchs[i][0], brchs[i][1])]

# Find patches
detection_map = 0 * bw
detection_patch = np.zeros([len(brchpt), 1000], dtype=np.int16)
thresh1, thresh2 = 0, 12
for i,brch in enumerate(brchpt):
    tmp_map = 0 * bw
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

#
# plt.imshow(detection_map+bw)
# plt.show()


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



fig = plt.gcf()
fig.set_size_inches(10, 10)

tmp_fig = np.copy(raw_img)
tmp_fig[detection_map>0, :] = 192.0/255.0
tmp_fig[skel>0, :] = 0
plt.imshow(tmp_fig)

plt.axis('off')
fig.tight_layout()
fig.subplots_adjust \
    (top=0.98,
bottom=0.04,
left=0.04,
right=0.985,
hspace=0.2,
wspace=0.2)
# plt.savefig( os.path.join(ROOT_PATH, filename+'_patch.png'), pad_inches=0.0, dpi=100)
plt.show()