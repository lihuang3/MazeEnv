"""
  gym id: Maze0522Env-v3
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


class Maze0522Env3(core.Env):
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
        self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = self.mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = 1.0 - self.mazeData
        self.freespace = 1.0 - self.freespaceData
        self.goal = np.array([22, 56])
        self.init_state = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _load_data(self, data_directory):
        mapname = 'map0522'
        filename = 'map0522v1'

        self.mazeData = np.loadtxt(data_directory + '/' + mapname + '.csv').astype(int)
        self.freespaceData = np.loadtxt(data_directory + '/' + mapname + '_freespace.csv').astype(int)
        self.costData = np.loadtxt(data_directory + '/' + filename + '_costmap.csv').astype(int)
        self.pgradData = np.loadtxt(data_directory + '/' + mapname + '_pgrad.csv').astype(int)
        self.flowstatsData = np.loadtxt(data_directory + '/' + mapname + '_flowstats.csv').astype(float)
        self.visitData = np.loadtxt(data_directory + '/' + mapname + '_visit.csv').astype(float)

        self.get_flowmap()
        self.get_flowstats()


    def _build_robot(self):
        self.internal_steps = 0
        self.delivery_rate_thresh = 0.7
        # ======================
        # For transfer learning only
        self.tflearn = False
        self.cur_robot = None
        # ======================
        row, col = np.where(np.logical_and(self.pgradData<=self.init_range, self.pgradData>0 ))
        self.reward_grad = np.zeros(40).astype(np.uint8)
        self.robot_num = 32  # len(row)
        self.doses = 4
        self.doses_remain = self.doses - 1
        self.dose_gap = 36
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

        # =====================
        # Transfer learning
        if self.tflearn:
            action = self.instructor()
            info = {'ac': action}
        # =====================

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
        elif delivery_rate >= self.delivery_rate_thresh:
            reward += 100 * (delivery_rate - self.delivery_rate_thresh)
            self.delivery_rate_thresh = delivery_rate
        elif delivery_rate >= 0.7  and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward += 4
        elif delivery_rate >= 0.6  and not self.reward_grad[2]:
            self.reward_grad[2] = 1
            reward += 4
        elif delivery_rate >= 0.5  and not self.reward_grad[3]:
            self.reward_grad[3] = 1
            reward += 4
        elif delivery_rate >= 0.4  and not self.reward_grad[4]:
            self.reward_grad[4] = 1
            reward += 2
        elif delivery_rate >= 0.3  and not self.reward_grad[5]:
            self.reward_grad[5] = 1
            reward += 2
        elif delivery_rate >= 0.2  and not self.reward_grad[6]:
            self.reward_grad[6] = 1
            reward += 2
        elif delivery_rate >= 0.1  and not self.reward_grad[7]:
            self.reward_grad[7] = 1
            reward += 1
        elif delivery_rate >= 0.05  and not self.reward_grad[8]:
            self.reward_grad[8] = 1
            reward += 1
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

        plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)
        plt.show(False)
        plt.pause(0.0001)

    def reset(self):
        return self._build_robot()

    def transfer_learning(self):
        if not self.tflearn:
            self.tflearn = True
            self.cur_robot = np.argmax(self.costData[self.loc[:, 0], self.loc[:, 1]])

    def instructor(self):

        _cost_to_goal = self.costData[self.loc[self.cur_robot, 0], self.loc[self.cur_robot, 1]]
        if _cost_to_goal < 5:
            self.cur_robot = np.argmax(self.costData[self.loc[:, 0], self.loc[:, 1]])
            _cost_to_goal = self.costData[self.loc[self.cur_robot, 0], self.loc[self.cur_robot, 1]]

        for i in range(-1, 2):
            for j in range(-1, 2):
                if not (i == 0 and j == 0):
                    new_pt = self.loc[self.cur_robot, :] + np.array([i, j]).astype(int)
                    new_cost = self.costData[new_pt[0], new_pt[1]]
                    if new_cost > 0 and new_cost < _cost_to_goal:
                        action = self.rev_action_map.get((i, j))
                        _cost_to_goal = np.copy(new_cost)

        return action

    def expert(self, robot_id):

        done, _ = self.get_reward()
        if robot_id is None or done or robot_id >= self.loc.shape[0]:
            return self.expert_restart_session()

        _cost_to_goal = self.costData[self.loc[robot_id, 0], self.loc[robot_id, 1]]
        if _cost_to_goal > 1:

            for i in range(-1, 2):
                for j in range(-1, 2):
                    if not (i == 0 and j == 0):
                        new_pt = self.loc[robot_id, :] + np.array([i, j]).astype(int)
                        new_cost = self.costData[new_pt[0], new_pt[1]]
                        if new_cost > 0 and new_cost < _cost_to_goal:
                            action = self.rev_action_map.get((i, j))
                            _cost_to_goal = np.copy(new_cost)

            return action, robot_id

        else:
            return self.expert_restart_session()

    def expert_restart_session(self):
        done, _ = self.get_reward()
        if done:
            self.reset()
        robot_id = np.argmax(self.costData[self.loc[:, 0], self.loc[:, 1]])
        return self.expert(robot_id)


def main(MazeEnv):
    env = MazeEnv()
    env.render()
    plt.pause(2)
    n_epochs = 10000
    robot_id = None
    steps = 0
    rewards = 0.0
    import time
    start = time.time()
    for i in range(n_epochs):
        if i % 200 == 0:
            now = time.time()
            print('step {} time elapse {}'.format(i, now - start))
            start = now
        steps += 1
        # next_action = np.random.randint(4,size = 1)
        next_action, robot_id = env.expert(robot_id)
        state_img, reward, done, _ = env._step(next_action)
        rewards += reward
        env.render()
        print('Step = %d, delivery_rate = %.2f, rewards = %.1f, reward = %.1f, done = %d' % (steps, env.delivery_rate, rewards, reward, done))
        if steps % 220 == 0:
            done = True

        if done:
            print('\n')

        if done:
            steps = 0
            rewards = 0.0
            plt.pause(2)
            env.reset()


if __name__ == '__main__':
    main(Maze0522Env3)


