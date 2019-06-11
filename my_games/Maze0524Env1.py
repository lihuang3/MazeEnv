"""
  gym id: Maze0524Env-v1
  Input: gray-scale images
  Reward: region range basis (easy)
  Render: gray-scale visualization
  Training: from scratch
  Open outlet!
"""

import matplotlib as mpl

mpl.use('TkAgg')
import numpy as np, random, sys, matplotlib.pyplot as plt, time, os
import gym
from gym import error, spaces, utils, core
from gym.utils import seeding

from time import sleep

plt.ion()


class Maze0524Env1(core.Env):
    def __init__(self):

        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.map_data_dir = dir_path + '/MapData'
        self.internal_steps = 0

        self.robot_marker = 150
        self.init_range = 20
        self.goal_range = 10

        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # {N, NE, E, SE, S, SW, W, NW}
        # self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (-1, 1), 3: (1, 1)}  # {up, down, left ,right}
        self.action_map = {0: (1, 0), 1: (1, 1), 2: (0, 1), 3: (-1, 1),
                           4: (-1, 0), 5: (-1, -1), 6: (0, -1), 7: (1, -1), 8: (0, 0)}
        self.rev_action_map = {(1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
                               (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7, (0, 0): 8}
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)
        self.goal = np.array([278, 64])

        self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = self.mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = 1.0 - self.mazeData
        self.freespace = 1.0 - self.freespaceData
        self.init_state = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _load_data(self, data_directory):
        mapname = 'map0524'
        filename = 'map0524v1'

        self.mazeData = np.loadtxt(data_directory + '/' + mapname + '.csv').astype(int)
        self.freespaceData = np.loadtxt(data_directory + '/' + mapname + '_freespace.csv').astype(int)
        self.costData = np.loadtxt(data_directory + '/' + filename + '_costmap.csv').astype(int)
        self.pgradData = np.loadtxt(data_directory + '/' + mapname + '_pgrad.csv').astype(int)
        self.flowstatsData = np.loadtxt(data_directory + '/' + mapname + '_flowstats.csv').astype(float)
        self.visitData = np.loadtxt(data_directory + '/' + mapname + '_visit.csv').astype(float)
        self.endpt_brch_map = np.loadtxt(data_directory + '/' + mapname + '_endpt_brch_map.csv').astype(int)
        self.detection_patch = np.loadtxt(data_directory + '/' + mapname + '_detect_patch.csv').astype(int)
        self.endpt_brch_control_map = np.loadtxt(data_directory + '/' + mapname + '_endpt_brch_control_map.csv').astype(int)
        self.get_flowmap()
        self.get_flowstats()
        self._init_control()

    def _build_robot(self):
        self.internal_steps = 0
        self.delivery_rate_thresh = 0.0

        # ======================
        # For transfer learning only
        self.tflearn = False
        self.cur_robot = None
        # ======================
        row, col = np.where(np.logical_and(self.pgradData<=self.init_range, self.pgradData>0 ))
        self.reward_grad = np.zeros(40).astype(np.uint8)
        self.robot_num = 64  # len(row)
        self.doses = 4
        self.doses_remain = self.doses - 1
        self.dose_gap = 100
        self.robot_num_orig = np.copy(self.robot_num)
        self.robot_num_prev = np.copy(self.robot_num)
        probs = self.visitData[row, col] / np.sum(self.visitData[row, col])
        self.robot = np.random.choice(row.shape[0], self.robot_num, p=probs)
        # self.robot = random.sample(range(row.shape[0]), self.robot_num)

        self.state = np.zeros(np.shape(self.mazeData)).astype(int)
        self.state_img = np.copy(self.state)
        self.loc = np.zeros([self.robot_num, 2]).astype(np.int32)
        for i in range(self.robot_num):
            self.loc[i, :] = row[self.robot[i]], col[self.robot[i]]
            self.state[row[self.robot[i]], col[self.robot[i]]] += self.robot_marker
            self.state_img[row[self.robot[i]] - 1:row[self.robot[i]] + 2,
            col[self.robot[i]] - 1:col[self.robot[i]] + 2] = self.robot_marker
        self.init_state = self.state
        self.init_state_img = self.state_img
        self.output_img = self.state_img + self.maze * 255
        return (np.expand_dims(self.output_img, axis=2))

    def _init_control(self):

        """
            self.endpt_brch_map: [endpt_row, endpt_col, len(brchs), \
            brchs1_row, brch1_col, brchs2_row, brch2_col, ... ]

            self.endpt_brch_control_map: [endpt_row, endpt_col, len(brchs), \
            dir1_row, dir1_col, dir2_row, dir2_col, ... ]

            self.detection_patch: [brch_row, brch_col, #(pixel in patch), \
            tl_row, br_row, tl_col, br_col, rows, cols]

        """
        res = np.sum(self.goal - self.endpt_brch_map[:,:2], axis=1)
        idx = np.squeeze(np.where(res == 0)[0])
        brch_size = int(self.endpt_brch_map[idx, 2])
        self.brch = np.reshape(self.endpt_brch_map[idx, 3:3+2*brch_size], [-1,2])
        self.brch_weights = np.ones(brch_size, dtype = np.float)
        # for i in range(brch_size):
        #     self.brch_weights[i] = (brch_size-i)/2.0
        self.brch_ctrl = np.reshape(self.endpt_brch_control_map[idx, 3:3+2*brch_size], [-1,2])
        patch_height = 1+np.max(self.detection_patch[:, 4] - self.detection_patch[:, 3])
        patch_width = 1+np.max(self.detection_patch[:, 6] - self.detection_patch[:, 5])
        self.patch = np.zeros([patch_height, patch_width, brch_size], dtype=np.int)
        self.tlpt = np.zeros([brch_size, 2], dtype=np.int)
        for i in range(brch_size):
            idx = np.where((self.brch[i, :] == self.detection_patch[:, :2]).all(axis=1))[0][0]
            num_pix = self.detection_patch[idx, 2]
            rows, cols = self.detection_patch[idx, 7:num_pix], self.detection_patch[idx, 7+num_pix:7+2*num_pix]
            tl_row, tl_col = self.detection_patch[idx, 3], self.detection_patch[idx, 5]
            self.tlpt[i,:] = [tl_row, tl_col]
            for row, col in zip(rows, cols):
                self.patch[row-tl_row,col-tl_col,i] = 1

    def get_costmap(self):
        return (np.expand_dims(self.costData, axis=2))

    def get_flowstats(self):
        dir_dict1 = [[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        dir_dict1_map1 = {0: [-1, 0], 1: [0, -1], 2: [1, 0], 3: [0, 1], 4: [1, 1], 5: [1, -1], 6: [-1, 1], 7: [-1, -1]}
        dir_dict1_map2 = {(-1, 0): 0, (0, -1): 1, (1, 0): 2, (0, 1): 3, (1, 1): 4, (1, -1): 5, (-1, 1): 6, (-1, -1): 7}

        self.flowstats = {}
        h, w = np.shape(self.mazeData)
        for i in range(np.shape(self.flowstatsData)[0]):
            dirs = []
            probs = []

            for j in range(8):
                if self.flowstatsData[i,j+1]>0:
                    dir = [-dir_dict1_map1[j][0], -dir_dict1_map1[j][1]]
                    dirs.append(dir)
                    probs.append(self.flowstatsData[i,j+1])
            probs = np.array(probs) / np.sum(probs)
            idx = int(self.flowstatsData[i, 0])
            self.flowstats[idx] = [dirs, probs]

    def get_flowmap(self):
        dir_dict1 = [[-1, 0], [0, -1], [1, 0], [0, 1], [1, 1], [1, -1], [-1, 1], [-1, -1]]
        rows, cols = np.where(self.pgradData>0)
        self.flow_dict = {}
        p1, p2 = 0.8, 0.2

        for row, col in zip(rows, cols):
            done = False
            cost = self.pgradData[row, col]
            candit = []
            candit0 = []

            for dir in dir_dict1:
                new_pt = np.array([row, col]) + np.array(dir)
                if self.pgradData[new_pt[0], new_pt[1]] == cost + 1:
                    candit.append(dir)
                elif self.pgradData[new_pt[0], new_pt[1]] == cost:
                    candit0.append(dir)

            a, b = len(candit), len(candit0)

            # Only use both cost+1 if available
            if a>0:
                prob = np.ones([a])/ (1.0*a)
                self.flow_dict[(row, col)] = [candit, prob]
            elif b>0:
                prob = np.ones([b])/ (1.0*b)
                self.flow_dict[(row, col)] = [candit0, prob]
            else:
                prob = [1.0]
                dir = [0,0]
                self.flow_dict[(row, col)] = [[dir], prob]

            # # Use both cost+ 1 and cost
            # prob = np.ones([a+b])
            # candit.extend(candit0)
            #
            # if a>0 and b>0:
            #     prob[:a] *= p1/a
            #     prob[a:] *= p2/b
            #     self.flow_dict[(row, col)] = [candit, prob]
            # elif a+b>0:
            #     prob *= 1.0/(a+b)
            #     self.flow_dict[(row, col)] = [candit, prob]
            # else:
            #     prob = [1.0]
            #     dir = [0,0]
            #     self.flow_dict[(row, col)] = [[dir], prob]

    def flow_step(self):
        dir = []
        h, w = np.shape(self.mazeData)
        for i in range(len(self.loc)):
            row, col = self.loc[i, :]
            idx = w*row + col
            try:
                item = self.flowstats[idx]
                dirs, probs = item[0], item[1]
                idx = np.random.choice(len(dirs), 1, p=probs)[0]
                dir.append(dirs[idx])
            except KeyError:
                item = self.flow_dict[(row, col)]
                dirs, prob = item[0], item[1]
                idx = np.random.choice(len(prob), 1, p=prob)[0]
                dir.append(dirs[idx])

        self.loc += np.array(dir)

    def respawn(self):
        row, col = np.where(np.logical_and(self.pgradData<=self.init_range, self.pgradData>0 ))
        probs = self.visitData[row, col] / np.sum(self.visitData[row, col])
        robot = np.random.choice(row.shape[0], self.robot_num, p=probs)
        # robot = random.sample(range(row.shape[0]), self.robot_num)
        loc = np.zeros([self.robot_num, 2]).astype(np.int32)
        for i in range(self.robot_num):
            loc[i, :] = row[robot[i]], col[robot[i]]

        self.loc = np.append(self.loc, loc, axis=0)

    def _step(self, action):
        info = {}
        self.internal_steps += 1
        if self.doses_remain>0 and (self.internal_steps % self.dose_gap == 1) and self.internal_steps>1:
            self.doses_remain -= 1
            self.respawn()

        for _ in range(3):
            self.flow_step()

        done, reward = self.get_reward()
        self.state_img *= 0

        for i in range(self.robot_num * (self.doses-self.doses_remain)):
            self.state_img[self.loc[i, 0] - 1:self.loc[i, 0] + 2,
            self.loc[i, 1] - 1:self.loc[i, 1] + 2] = self.robot_marker

        self.output_img = self.state_img + self.maze * 255
        return (np.expand_dims(self.output_img, axis=2), reward, done, info)

    def step(self, action):

        info = {}
        self.internal_steps += 1
        if self.doses_remain>0 and (self.internal_steps % self.dose_gap == 1) and self.internal_steps>1:
            self.doses_remain -= 1
            self.respawn()

        for _ in range(3):
            self.flow_step()

        dy, dx = self.action_map[action]

        prev_loc = np.copy(self.loc)
        self.loc = np.add(self.loc, np.array([dy, dx]))
        # escaped = np.where(self.outlet[self.loc[:, 0], self.loc[:, 1]] == 2.0)
        collision = np.where(self.freespace[self.loc[:, 0], self.loc[:, 1]] == 1.0)
        # escaped = np.where(self.outletData[self.loc[collision, 0], self.loc[collision, 1]] == 2.0)[1]
        self.loc[collision, :] = prev_loc[collision, :]
        # if len(escaped) > 0 and (self.robot_num - len(escaped) > 1):
        #     self.loc = np.delete(self.loc, collision[0][escaped], axis=0)
        #     self.robot_num = self.loc.shape[0]

        self.state_img *= 0

        for i in range(self.robot_num * (self.doses-self.doses_remain) ):
            self.state_img[self.loc[i, 0] - 1:self.loc[i, 0] + 2,
            self.loc[i, 1] - 1:self.loc[i, 1] + 2] = self.robot_marker

        self.output_img = self.state_img + self.maze * 255

        done, reward = self.get_reward()

        return (np.expand_dims(self.output_img, axis=2), reward, done, info)

    def get_reward(self):

        cost_arr = self.costData[self.loc[:, 0], self.loc[:, 1]]
        cost_arr = cost_arr[cost_arr<=self.goal_range]
        self.delivery_rate = delivery_rate = cost_arr.shape[0]/(self.robot_num * self.doses)

        done = False
        reward = -0.05
        if delivery_rate >= 0.90 :
            done = True
            reward += 100
            return done, reward
        elif delivery_rate >= 0.025 + self.delivery_rate_thresh:
            reward += 40 * (delivery_rate - self.delivery_rate_thresh)
            self.delivery_rate_thresh = np.copy(delivery_rate)
        # if reward>0:
        #     print(reward)
        # if delivery_rate >= 0.5  and not self.reward_grad[3]:
        #     self.reward_grad[3] = 1
        #     reward += 4
        # elif delivery_rate >= 0.4  and not self.reward_grad[4]:
        #     self.reward_grad[4] = 1
        #     reward += 2
        # elif delivery_rate >= 0.3  and not self.reward_grad[5]:
        #     self.reward_grad[5] = 1
        #     reward += 2
        # elif delivery_rate >= 0.2  and not self.reward_grad[6]:
        #     self.reward_grad[6] = 1
        #     reward += 2
        # elif delivery_rate >= 0.1  and not self.reward_grad[7]:
        #     self.reward_grad[7] = 1
        #     reward += 1
        # elif delivery_rate >= 0.05  and not self.reward_grad[8]:
        #     self.reward_grad[8] = 1
        #     reward += 1
        return done, reward

    def render(self, mode='human'):
        plt.gcf().clear()

        render_image = np.copy(0 * self.maze).astype(np.int16)
        for i in range(self.robot_num * (self.doses-self.doses_remain) ):
            render_image[self.loc[i, 0] - 1:self.loc[i, 0] + 2,
            self.loc[i, 1] - 1:self.loc[i, 1] + 2] += self.robot_marker

        row, col = np.nonzero(render_image)
        min_robots = 150.
        max_robots = float(np.max(render_image))
        # rgb_render_image = np.stack((render_image+self.maze*255,)*3, -1)
        rgb_render_image = np.stack(
            (render_image + self.maze * 128, render_image + self.maze * 228, render_image + self.maze * 255), -1)
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


        circle = plt.Circle((self.goal[1], self.goal[0]), 10, linestyle='--', color='red', linewidth=1, fill=False)
        plt.gcf().gca().add_artist(circle)

        plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)
        plt.show(False)
        plt.pause(0.0001)

    def reset(self):
        return self._build_robot()

    def expert(self):
        patch_height, patch_width, patch_depth = self.patch.shape
        action_weights = np.ones(patch_depth)
        for i in range(self.brch.shape[0]):
            action_weights[i] = self.brch_weights[i]* np.sum(np.multiply(self.patch[:,:,i],
                                      self.state_img[self.tlpt[i,0]:self.tlpt[i,0]+patch_height,
                                      self.tlpt[i,1]:self.tlpt[i,1]+patch_width]))/self.robot_marker
        # print( np.exp(action_weights)/sum(np.exp(action_weights)))
        selected_brch = np.argmax(action_weights)
        if action_weights[selected_brch] == 0:
            action = self.rev_action_map[(0, 0)]
        else:
            dir = (self.brch_ctrl[selected_brch][0],self.brch_ctrl[selected_brch][1])
            action = self.rev_action_map[dir]
        return action

def DFS(weights, cur_brch, weight_dict, weights_set):
    if cur_brch >= len(weights):
        weights_set.append(np.copy(weights))
        return
    for i in range(len(weight_dict)):
        weights[cur_brch] = weight_dict[i]
        DFS(weights, cur_brch+1, weight_dict, weights_set)


def _main(MazeEnv, args):
    import datetime
    from mpi4py import MPI

    env = MazeEnv()

    # env.render()
    episode = 10
    steps = 0
    rewards = 0
    import time
    weight_dict = [1, 2, 4, 8]
    brch_size = env.brch_weights.shape[0]
    weights = [1] * brch_size
    weights_set = []
    DFS(weights=weights, cur_brch=0, weight_dict=weight_dict, weights_set=weights_set)

    # if MPI.COMM_WORLD.Get_size() > 1:
    num_workers = MPI.COMM_WORLD.Get_size()
    assert num_workers % len(weights_set)
    my_rank = int(MPI.COMM_WORLD.Get_rank())
    my_portion = int(len(weights_set) / num_workers)
    weights_set = weights_set[ my_rank*my_portion:(my_rank+1)*my_portion ]
    sendbuf = -1.0 * np.ones([my_portion, brch_size+2])
    recvbuf = None
    if my_rank == 0:
        recvbuf = np.empty([num_workers, my_portion, brch_size+2], dtype=np.float)
    start = time.time()
    for cnt, env.brch_weights in enumerate(weights_set):
        delivery = []
        steps = 0
        for i in range(episode):
            done = False
            while not done:
                steps += 1
                next_action = env.expert()
                _, reward, done, _ = env.step(next_action)
                rewards += reward
                if steps > 0 and steps % args.nsteps == 0:
                    delivery.append(env.delivery_rate)
                    done = True

                if done:
                    steps = 0
                    rewards = 0
                    env.reset()
        mean = 100.0 * np.mean(delivery)
        std = 100.0 * np.std(delivery)
        sendbuf[cnt, :2] = [mean, std]
        sendbuf[cnt, 2:] = env.brch_weights
        time_left = str(datetime.timedelta(seconds=(time.time() - start) * (len(weights_set) - cnt - 1) / (cnt + 1) ))
        print('%d/%d'%(1+cnt, len(weights_set)), 'worker_%d'%(my_rank), 'time left:', time_left[:-7], 'weights=',env.brch_weights, ' deli mean=%.2f'%(mean), '% ', ' deli std=%.2f'%(std),'%')
        sys.stdout.flush()

    MPI.COMM_WORLD.Gather(sendbuf, recvbuf, root=0)
    if my_rank == 0:
        recvbuf = np.reshape(recvbuf, [-1, brch_size+2])
        assert(recvbuf.min()>=0)
        sorted_res = recvbuf[recvbuf[:,0].argsort()]
        print(sorted_res[-32:,:])
        weight_dir = os.path.abspath('./weights')
        filename = args.env + '_weights_candate.csv'
        np.savetxt(os.path.join(weight_dir, filename), sorted_res[-32:,2:], fmt='%3i')

def finetune(MazeEnv, args):
    import datetime
    from mpi4py import MPI

    env = MazeEnv()

    weight_dir = os.path.abspath('./weights')
    filename = args.env + '_weights_candate.csv'
    weights_set = np.loadtxt(os.path.join(weight_dir, filename)).astype(int)
    episode = 32
    steps = 0
    rewards = 0
    import time
    brch_size = env.brch_weights.shape[0]

    num_workers = MPI.COMM_WORLD.Get_size()
    assert weights_set.shape[0] % num_workers == 0
    my_rank = int(MPI.COMM_WORLD.Get_rank())
    my_portion = int(weights_set.shape[0] / num_workers)
    weights_set = weights_set[ my_rank*my_portion:(my_rank+1)*my_portion ]
    sendbuf = -1.0 * np.ones([my_portion, brch_size+2])
    recvbuf = None
    if my_rank == 0:
        recvbuf = np.empty([num_workers, my_portion, brch_size+2], dtype=np.float)
    start = time.time()
    for cnt in range(weights_set.shape[0]):
        delivery = []
        steps = 0
        env.brch_weights = weights_set[cnt, :]
        for i in range(episode):
            done = False
            while not done:
                steps += 1
                next_action = env.expert()
                _, reward, done, _ = env.step(next_action)
                rewards += reward
                if steps > 0 and steps % args.nsteps == 0:
                    delivery.append(env.delivery_rate)
                    done = True

                if done:
                    steps = 0
                    rewards = 0
                    env.reset()
        mean = 100.0 * np.mean(delivery)
        std = 100.0 * np.std(delivery)
        sendbuf[cnt, :2] = [mean, std]
        sendbuf[cnt, 2:] = env.brch_weights
        time_left = str(datetime.timedelta(seconds=(time.time() - start) * (len(weights_set) - cnt - 1) / (cnt + 1) ))
        print('%d/%d'%(1+cnt, len(weights_set)), 'worker_%d'%(my_rank), 'time left:', time_left[:-7], 'weights=',env.brch_weights, ' deli mean=%.2f'%(mean), '% ', ' deli std=%.2f'%(std),'%')
        sys.stdout.flush()

    MPI.COMM_WORLD.Gather(sendbuf, recvbuf, root=0)
    if my_rank == 0:
        recvbuf = np.reshape(recvbuf, [-1, brch_size+2])
        assert(recvbuf.min()>=0)
        sorted_res = recvbuf[recvbuf[:,0].argsort()]
        print(sorted_res)

def main(MazeEnv, args):
    env = MazeEnv()
    env.render()
    steps = 0
    rewards = 0
    import time
    start = time.time()
    env.brch_weights = args.weights

    steps = 0
    while 1:
        steps += 1
        next_action = env.expert()
        _, reward, done, _ = env.step(next_action)
        rewards += reward
        env.render()
        print('Step = %d, deli = %.2f, rew = %.2f, done = %d' % (steps, env.delivery_rate, rewards, done))
        if steps > 0 and steps % args.nsteps == 0:
            print('deli = ', 100.0 * env.delivery_rate, '%', 'rew = %.2f'%(rewards))
            done = True
            time.sleep(2)
        if done:
            steps = 0
            rewards = 0
            env.reset()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mode', type=str, default='test', choices=['train', 'test', 'fitu'])
    parser.add_argument('--env', type=str, default='Maze0524Env1')
    parser.add_argument('--nsteps', type=int, default=500)
    parser.add_argument('--weights', type=list, default=[1,1, 1, 1, 1])

    args = parser.parse_args()

    maze = Maze0524Env1
    if args.mode == 'test':
        main(maze, args)
    elif args.mode == 'train':
        _main(maze, args)
    elif args.mode == 'fitu':
        finetune(maze, args)
    else:
        raise NotImplementedError
