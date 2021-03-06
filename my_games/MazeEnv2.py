"""
  gym id: MazeEnv-v2
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


class MazeEnv2(core.Env):
    def __init__(self):
        global mazeData, costData, centerline, freespace, mazeHeight, mazeWidth, robot_marker
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.map_data_dir = dir_path+'/MapData'
        robot_marker = 150
        self.goal_range = 10
        self.actions = [1, 2, 3, 4] # {up, down, left ,right}
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, 1), 3: (0, -1)}  # {up, down, left ,right}
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)
        mazeData, costData, freespace = self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.freespace = np.ones((mazeHeight, mazeWidth))-freespace
        # self.goal = np.array([73, 10])
        self.goal = np.array([31, 52])
        self.init_state = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _load_data(self, data_directory):
        filename = 'scaled_maze7'

        mazeData = np.loadtxt(data_directory + '/'+filename+'.csv').astype(int)
        freespace = np.loadtxt(data_directory + '/'+filename+'_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/' +filename+ '_costmap1.csv').astype(int)
        return mazeData, costData, freespace

    def _build_robot(self):

        row, col = np.nonzero(freespace)
        self.reward_grad = np.zeros(40).astype(np.uint8)
        self.robot_num = 64 #len(row)
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
        
    def step(self,action):
        info = {}

        dy, dx = self.action_map[action]
        prev_loc = np.copy(self.loc)
        self.loc = np.add(self.loc, np.array([dy, dx]))
        collision = np.where(self.freespace[self.loc[:, 0], self.loc[:, 1]] == 1.0)
        self.loc[collision, :] = prev_loc[collision, :]
        self.state_img  *= 0 # np.zeros([mazeHeight,mazeWidth])
        for i in range(self.robot_num):
            self.state_img[self.loc[i,0]-1:self.loc[i,0]+1, self.loc[i,1]-1:self.loc[i,1]+1] = robot_marker
        self.output_img = self.state_img + self.maze*255
        done, reward = self.get_reward()
        return(np.expand_dims(self.output_img,axis=2),reward,done,info)

    def get_reward(self):
        cost_to_go = np.sum(costData[self.loc[:, 0], self.loc[:, 1]])
        max_cost_agent = np.max(costData[self.loc[:, 0], self.loc[:, 1]])

        done = False
        reward = -.1

        if max_cost_agent <= self.goal_range:
            done = True
            reward = 100.0
        elif max_cost_agent <= 2*self.goal_range and not self.reward_grad[0]:
            self.reward_grad[0] = 1
            reward = 8
        elif max_cost_agent <= 3*self.goal_range and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward = 4
        elif max_cost_agent <= 4*self.goal_range and not self.reward_grad[2]:
            self.reward_grad[2] = 1
            reward = 4
        elif max_cost_agent <= 5*self.goal_range and not self.reward_grad[3]:
            self.reward_grad[3] = 1
            reward = 2
        elif max_cost_agent <= 6*self.goal_range and not self.reward_grad[4]:
            self.reward_grad[4] = 1
            reward = 2

        if cost_to_go <= self.goal_range * self.robot_num and not self.reward_grad[20]:
            self.reward_grad[20] = 1
            reward = 8
        elif cost_to_go <= 2*self.goal_range * self.robot_num and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward = 4
        elif cost_to_go <= 3*self.goal_range * self.robot_num  and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward = 2
        elif cost_to_go <= 4*self.goal_range * self.robot_num  and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward = 2
        elif cost_to_go <= 5*self.goal_range * self.robot_num  and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward = 2

        return done, reward

    def render(self, mode = 'human'):
        # plt.imshow(self.state_img + self.maze*255, vmin=0, vmax=255)
        plt.imshow(self.output_img)
        plt.show(False)
        plt.pause(0.0001)
        plt.gcf().clear()


    def reset(self):
        return self._build_robot()


    def expert(self, robot_loc):

        _cost_to_goal = np.sum(self.state*costData/robot_marker)
        if not len(robot_loc) or _cost_to_goal<=self.robot_num*self.goal_range:
            return self.expert_restart_session()

        _cost_to_goal = costData[robot_loc[0], robot_loc[1]]
        if _cost_to_goal >1:
            _cost_to_goal -= 1

            for i in range(4):
                new_pt = robot_loc + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                if (costData[new_pt[0], new_pt[1]] == _cost_to_goal):
                    if np.absolute(new_pt - robot_loc)[0]:
                        action = (np.amax([0, (new_pt - robot_loc)[0]]))
                    if np.absolute(new_pt - robot_loc)[1]:
                        action = (np.amax([2, 2 + (new_pt - robot_loc)[1]]))
                    robot_loc = new_pt
                    return action, robot_loc
                elif(costData[new_pt[0], new_pt[1]] == _cost_to_goal+1):
                    if np.absolute(new_pt - robot_loc)[0]:
                        action = (np.amax([0, (new_pt - robot_loc)[0]]))
                    if np.absolute(new_pt - robot_loc)[1]:
                        action = (np.amax([2, 2 + (new_pt - robot_loc)[1]]))

                    robot_loc = new_pt
            return action, robot_loc


        else:
            return self.expert_restart_session()

    def expert_restart_session(self):
        if (np.sum(self.state*costData/robot_marker)<=self.robot_num*self.goal_range):
            self.reset()
        robot_loc = np.unravel_index(np.argmax((self.state > 0) * costData), self.state.shape)
        return self.expert(robot_loc)

def main(MazeEnv):
    env = MazeEnv()
    env.render()
    plt.pause(2)
    n_epochs = 10000
    robot_loc =[]
    steps  = 0
    rewards = 0.0
    for i in range(n_epochs):
        steps+=1
        #next_action = np.random.randint(4,size = 1)
        next_action, robot_loc = env.expert(robot_loc)
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
    main(MazeEnv2)