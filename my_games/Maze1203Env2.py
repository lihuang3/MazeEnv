"""
  gym id: Maze1203Env-v2
  Input: gray-scale images
  Reward: region range basis (easy)
  Render: gray-scale visualization
  Training: from scratch
"""

import matplotlib as mpl
mpl.use('TkAgg')
import numpy as np, random, sys, matplotlib.pyplot as plt, time, os
import gym
from gym import error, spaces, utils, core
from gym.utils import seeding

from time import sleep
plt.ion()

from scipy.stats import gaussian_kde
ROOT_DIR = os.path.abspath("./")

class Maze1203Env2(core.Env):
    def __init__(self):
        global mazeData, costData, freespace, mazeHeight, mazeWidth, robot_marker
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.map_data_dir = dir_path + '/MapData'
        self.save_frame = True
        self.loc_hist = [[],[],[],[]]
        self.internal_steps = 0
        os.makedirs(ROOT_DIR+'/frames', exist_ok=True)

        robot_marker = 150
        self.goal_range = 10
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7] # {N, NE, E, SE, S, SW, W, NW}
        # self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (-1, 1), 3: (1, 1)}  # {up, down, left ,right}
        self.action_map = {0: (1, 0),  1: (1, 1),  2: (0, 1), 3: (-1, 1),
                           4: (-1, 0), 5: (-1,-1), 6: (0,-1), 7: (1, -1)}
        self.rev_action_map = {(1, 0):0, (1, 1):1, (0, 1):2, (-1, 1):3,
                           (-1, 0):4, (-1, -1):5, (0, -1):6, (1, -1):7}
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)
        mazeData, costData, freespace = self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.freespace = np.ones((mazeHeight, mazeWidth))-freespace

        self.goal = np.array([130, 61])
        self.init_state = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _load_data(self, data_directory):
        filename = 'map1203'
        RGBmap = 'map1203'
        self.raw_img = plt.imread(data_directory + '/'+RGBmap+'.png')
        mazeData = np.loadtxt(data_directory + '/'+filename+'.csv').astype(int)
        freespace = np.loadtxt(data_directory + '/'+filename+'_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/' +filename+ '_costmap.csv').astype(int)
        return mazeData, costData, freespace

    def _build_robot(self):
        self.internal_steps = 0

        row, col = np.nonzero(freespace)
        self.reward_grad = np.zeros(40).astype(np.uint8)
        self.robot_num = 1024 #len(row)
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        self.state_img = np.copy(self.state)
        self.loc = np.zeros([self.robot_num,2]).astype(np.int32)
        for i in range(self.robot_num):
            self.loc[i, :] = row[self.robot[i]], col[self.robot[i]]
            self.state[row[self.robot[i]], col[self.robot[i]]] += robot_marker
            self.state_img[row[self.robot[i]]-1:row[self.robot[i]]+2,
                col[self.robot[i]]-1:col[self.robot[i]]+2] = robot_marker
        self.init_state = self.state
        self.init_state_img = self.state_img
        self.output_img = self.state_img + self.maze * 255
        return (np.expand_dims(self.output_img, axis=2))

    def get_costmap(self):
        return (np.expand_dims(costData, axis=2))
        
    def step(self, action):

        info = {}

        self.action = action
        # For sticky action rendering usage

        self.loc_hist[self.internal_steps % 4] = self.loc
        self.internal_steps += 1

        dy, dx = self.action_map[action]

        prev_loc = np.copy(self.loc)
        self.loc = np.add(self.loc, np.array([dy, dx]))
        collision = np.where(self.freespace[self.loc[:, 0], self.loc[:, 1]] == 1.0)
        self.loc[collision, :] = prev_loc[collision, :]

        self.state_img  *= 0 # np.zeros([mazeHeight,mazeWidth])

        for i in range(self.robot_num):
            self.state_img[self.loc[i,0]-1:self.loc[i,0]+2, self.loc[i,1]-1:self.loc[i,1]+2] = robot_marker

        self.output_img = self.state_img + self.maze*255

        done, reward = self.get_reward()

        return(np.expand_dims(self.output_img,axis=2),reward,done,info)

    def get_reward(self):

        cost_arr = costData[self.loc[:, 0], self.loc[:, 1]]
        cost_to_go = np.sum(cost_arr)
        cost_arr = cost_arr[cost_arr<=self.goal_range]
        self.delivery_rate = delivery_rate = cost_arr.shape[0]/ self.robot_num

        max_cost_agent = np.max(costData[self.loc[:, 0], self.loc[:, 1]])

        done = False
        reward = -0.1

        if max_cost_agent <= self.goal_range:
            done = True
            reward += 200
            return done, reward
        elif max_cost_agent <= 2 * self.goal_range and not self.reward_grad[0]:
            self.reward_grad[0] = 1
            reward += 8
        elif max_cost_agent <= 4 * self.goal_range and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward += 8
        elif max_cost_agent <= 8 * self.goal_range and not self.reward_grad[2]:
            self.reward_grad[2] = 1
            reward += 4
        elif max_cost_agent <= 12 * self.goal_range and not self.reward_grad[3]:
            self.reward_grad[3] = 1
            reward += 4
        elif max_cost_agent <= 16 * self.goal_range and not self.reward_grad[4]:
            self.reward_grad[4] = 1
            reward += 4
        elif max_cost_agent <= 20 * self.goal_range and not self.reward_grad[5]:
            self.reward_grad[5] = 1
            reward += 4


        if cost_to_go <= self.goal_range * self.robot_num and not self.reward_grad[20]:
            self.reward_grad[20] = 1
            reward += 4
        elif cost_to_go <= 2 * self.goal_range * self.robot_num and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward += 4
        elif cost_to_go <= 4 * self.goal_range * self.robot_num and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward += 4
        elif cost_to_go <= 8 * self.goal_range * self.robot_num and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward += 2
        elif cost_to_go <= 12 * self.goal_range * self.robot_num and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward += 2
        elif cost_to_go <= 16 * self.goal_range * self.robot_num and not self.reward_grad[25]:
            self.reward_grad[25] = 1
            reward += 2
        elif cost_to_go <= 20 * self.goal_range * self.robot_num and not self.reward_grad[26]:
            self.reward_grad[26] = 1
            reward += 2

        return done, reward

    def render2(self, mode = 'human'):
        plt.gcf().clear()

        render_image = np.copy(0*self.maze).astype(np.int16)
        for i in range(self.robot_num):
            render_image[self.loc[i,0]-1:self.loc[i,0]+1, self.loc[i,1]-1:self.loc[i,1]+1] += robot_marker

        row, col = np.nonzero(render_image)
        min_robots = 150.
        max_robots = float(np.max(render_image))
        # rgb_render_image = np.stack((render_image+self.maze*255,)*3, -1)
        rgb_render_image = np.stack((render_image+self.maze*128, render_image+self.maze*228, render_image+self.maze*255 ), -1)
        rgb_render_image[rgb_render_image[:,:,:]==0]=255

        for i in range(row.shape[0]):
            value = render_image[row[i],col[i]]
            ratio = 0.4 + 0.5 * max(value - min_robots, 0) / (max_robots - min_robots)
            ratio = min(0.9, max(0.4, ratio))
            b = 180
            g = 180 * (1 - ratio)
            r = 180 * (1 - ratio)

            for j, rgb in enumerate([r,g,b]):
                rgb_render_image[row[i], col[i], j] = np.uint8(rgb)

        plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)
        plt.show(False)
        plt.pause(0.0001)

    def render_config(self, loc, dot_size=15):
        # Load high-resolution
        plt.imshow(self.raw_img)
        h, w, _ = self.raw_img.shape
        h1, w1 = self.maze.shape
        # Draw goal range
        circle = plt.Circle((float(w)/w1*self.goal[1], float(h)/h1*self.goal[0]), 40, linestyle='-', color='red', linewidth=2, fill=False)
        plt.gcf().gca().add_artist(circle)

        text_str = '%.1f'%(100*self.delivery_rate) + ' %'
        plt.text(w-150, 50, text_str, fontsize=16)

        dir = self.action_map[self.action]
        offset = 80
        plt.arrow(offset, h-offset, 30*dir[1], 30*dir[0], head_width=10, head_length=10, fc='k', ec='k')
        # Configure robot density
        ys, xs =float(h)/h1*loc[:,0], float(w)/w1*loc[:,1]
        xy = np.vstack([xs, ys])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        xs, ys, z = xs[idx], ys[idx], z[idx]
        cmap = mpl.cm.get_cmap('Blues')
        upperZ = max(z)
        lowerZ = min(z)
        norm = mpl.colors.Normalize(vmin = lowerZ, vmax=upperZ)
        z[z<0.7*upperZ] = 0.7*upperZ
        colors = [cmap(norm(value)) for value in z]
        # Draw robots
        plt.scatter(xs, ys, c=colors, s=dot_size, edgecolor='')

    def render(self, mode = 'human'):
        for i in range(4):
            loc = self.loc_hist[i]
            plt.gcf().clear()

            self.render_config(loc)
            if self.save_frame:
                h, w, _ = self.raw_img.shape
                fig = plt.gcf()
                fig.set_size_inches(7.5, h*7.5/w)
                filename = ROOT_DIR + '/frames/' + 'fig' + '%05d' % (1 + len(os.listdir(ROOT_DIR + '/frames'))) + '.png'
                plt.axis('off')
                fig.tight_layout()
                fig.subplots_adjust\
                        (top=1.0,
                        bottom=0.0,
                        left=0.0,
                        right=1.0,
                        hspace=0.0,
                        wspace=0.0)
                plt.savefig(filename, pad_inches=0.0, dpi=100)
            else:
                plt.show(False)
                plt.pause(0.0001)


    def reset(self):
        return self._build_robot()



    def expert(self, robot_id):

        done, _ = self.get_reward()

        if robot_id is None or done:
            return self.expert_restart_session()

        _cost_to_goal = costData[self.loc[robot_id,0], self.loc[robot_id,1]]
        if _cost_to_goal >1:

            for i in range(-1,2):
                for j in range(-1,2):
                    if not (i==0 and j==0):
                        new_pt = self.loc[robot_id, :] + np.array([i,j]).astype(int)
                        new_cost = costData[new_pt[0], new_pt[1]]
                        if new_cost>0 and new_cost < _cost_to_goal:
                            action = self.rev_action_map.get((i,j))
                            _cost_to_goal = np.copy(new_cost)

            return action, robot_id

        else:
            return self.expert_restart_session()

    def expert_restart_session(self):
        done, _ = self.get_reward()
        if done:
            self.reset()
        robot_id = np.argmax(costData[self.loc[:, 0], self.loc[:, 1]])
        return self.expert(robot_id)

def main(MazeEnv):
    env = MazeEnv()
    env.render()
    plt.pause(2)
    n_epochs = 10000
    robot_id = None
    steps  = 0
    rewards = 0.0
    import time
    start = time.time()
    for i in range(n_epochs):
        if i%50 ==0:
            now = time.time()
            print('step {} time elapse {}'.format(i, now-start))
            start = now
        steps+=1
        #next_action = np.random.randint(4,size = 1)
        next_action, robot_id = env.expert(robot_id)
        state_img,reward, done, _ = env.step(next_action)
        rewards +=reward
        env.render()
        print('Step = %d, rewards = %.1f, reward = %.1f, done = %d'%(steps, rewards, reward, done), end='\r')
        if done:
            print('\n')


        if done:
            steps = 0
            rewards = 0.0
            plt.pause(2)
            env.reset()


if __name__ == '__main__':
    main(Maze1203Env2)


