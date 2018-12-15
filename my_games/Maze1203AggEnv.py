"""
  gym id: Maze1203AggEnv-v0
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


class Maze1203AggEnv(core.Env):
    def __init__(self):
        global mazeData, costData, freespace, mazeHeight, mazeWidth, robot_marker
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.map_data_dir = dir_path+'/MapData'

        robot_marker = 150
        self.goal_range = 15
        self.actions = [1, 2, 3, 4] # {up, down, left ,right}
        self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (-1, 1), 3: (1, 1)}  # {up, down, left ,right}
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0,-1), 3: (0,1)}
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)
        mazeData, costData, freespace = self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.freespace = np.ones((mazeHeight, mazeWidth))-freespace
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _load_data(self, data_directory):
        filename = 'map1203'
        mazeData = np.loadtxt(data_directory + '/'+filename+'.csv').astype(int)
        freespace = np.loadtxt(data_directory + '/'+filename+'_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/' +filename+ '_costmap.csv').astype(int)
        return mazeData, costData, freespace

    def _build_robot(self):
        row, col = np.nonzero(freespace)
        self.reward_grad = np.zeros(10).astype(np.uint8)
        self.robot_num = 128 #len(row)
        self.robot = random.sample(range(row.shape[0]), self.robot_num)
        self.state = np.zeros(np.shape(mazeData)).astype(int)
        self.state_img = np.copy(self.state)
        self.loc = np.zeros([self.robot_num,2]).astype(np.int32)
        for i in range(self.robot_num):
            self.loc[i, :] = row[self.robot[i]], col[self.robot[i]]
            self.state[row[self.robot[i]], col[self.robot[i]]] += robot_marker
            self.state_img[row[self.robot[i]]-1:row[self.robot[i]]+2,
                col[self.robot[i]]-1:col[self.robot[i]]+2] = robot_marker*np.ones([3,3])

        self.agg_area_init = np.sum(self.state_img > 0)
        self.agg_area = np.copy(self.agg_area_init)
        self.output_img = self.state_img + self.maze * 255

        return (np.expand_dims(self.output_img, axis=2))

    def step(self, action):
        dy, dx = self.action_map[action]
        prev_loc = np.copy(self.loc)
        self.loc = np.add(self.loc, np.array([dy, dx]))
        collision = np.where(self.freespace[self.loc[:, 0], self.loc[:, 1]] == 1.0)
        self.loc[collision, :] = prev_loc[collision, :]

        self.state_img  *= 0 # np.zeros([mazeHeight,mazeWidth])

        for i in range(self.robot_num):
            self.state_img[self.loc[i,0]-1:self.loc[i,0]+2, self.loc[i,1]-1:self.loc[i,1]+2] = robot_marker * np.ones([3, 3])

        self.output_img = self.state_img + self.maze*255

        self.agg_area = np.sum(self.state_img > 0)


        done = False
        reward = -.1

        if self.agg_area <= self.goal_range:
          done = True
          reward = 200
        elif self.agg_area  <= 2 * self.goal_range and not self.reward_grad[0]:
          self.reward_grad[0] = 1
          reward = 16
        elif self.agg_area  <= 4 * self.goal_range and not self.reward_grad[1]:
          self.reward_grad[1] = 1
          reward = 8
        elif self.agg_area  <= 6 * self.goal_range and not self.reward_grad[2]:
          self.reward_grad[2] = 1
          reward = 8
        elif self.agg_area  <= 8 * self.goal_range and not self.reward_grad[3]:
          self.reward_grad[3] = 1
          reward = 4

        if self.agg_area  <= self.agg_area_init * 0.1 and not self.reward_grad[4]:
          done = False
          self.reward_grad[4] = 1
          reward = 8
        elif self.agg_area  <= 2 * self.agg_area_init * 0.1 and not self.reward_grad[5]:
          done = False
          self.reward_grad[5] = 1
          reward = 4
        elif self.agg_area  <= 4 * self.agg_area_init * 0.1 and not self.reward_grad[6]:
          done = False
          self.reward_grad[6] = 1
          reward = 4

        info = {}

        return(np.expand_dims(self.output_img,axis=2),reward,done,info)

    def _step(self,action):

        next_direction, next_axis = self.action_dict[action]

        next_state = np.roll(self.state, next_direction, axis=next_axis)

        # Collision check
        collision = np.logical_and(next_state, self.freespace)*next_state

        next_state *= np.logical_xor(next_state, self.freespace)

        # Move robots in the obstacle area back to previous grids and obtain the next state
        ## Case 1: overlapping with population index
        next_state += np.roll(collision, -next_direction, axis=next_axis)
        ## Case 2: overlapping w/o population index (0: no robot; 1: robot(s) exits)
        # next_state = np.logical_or(np.roll(collision, -next_direction, axis=next_axis), next_state).astype(int)

        # next_state *= robot_marker   # Mark robot with intensity 150

        row, col = np.nonzero(next_state)

        self.state_img  *= 0 # np.zeros([mazeHeight,mazeWidth])

        for i in range(row.shape[0]):
            self.state_img[row[i]-2:row[i]+3, col[i]-2:col[i]+3] = robot_marker * np.ones([5, 5])

        self.state = next_state

        self.output_img = self.state_img + self.maze*255

        state_cost_matrix = self.state * costData/ robot_marker
        cost_to_go = np.sum(state_cost_matrix)

        done = False
        reward = -.1

        if cost_to_go <= self.goal_range * self.robot_num:
            done = True
            reward = 100.0
        elif cost_to_go <= 2*self.goal_range * self.robot_num and not self.reward_grad[0]:
            self.reward_grad[0] = 1
            reward = 20.0
        elif cost_to_go <= 3*self.goal_range and not self.reward_grad[1]:
            self.reward_grad[1] = 1
            reward = 5.0

        info = {}

        return(np.expand_dims(self.output_img,axis=2),reward,done,info)

    def render(self, mode = 'human'):
        # plt.gcf().clear()
        # plt.imshow(self.output_img)
        # plt.show(False)
        # plt.pause(0.0001)

        plt.gcf().clear()

        render_image = np.copy(0*self.maze).astype(np.int16)
        for i in range(self.robot_num):
            render_image[self.loc[i,0]-1:self.loc[i,0]+2, self.loc[i,1]-1:self.loc[i,1]+2] += robot_marker

        row, col = np.nonzero(render_image)
        min_robots = 150.
        max_robots = float(np.max(render_image))
        rgb_render_image = np.stack((render_image+self.maze*255,)*3, -1)



        for i in range(row.shape[0]):
            value = render_image[row[i],col[i]]
            ratio = 0.4 + 0.5 * max(value - min_robots, 0) / (max_robots - min_robots)
            ratio = min(0.9, max(0.4, ratio))
            b = 255
            g = 255 * (1 - ratio)
            r = 255 * (1 - ratio)

            for j, rgb in enumerate([r,g,b]):
                rgb_render_image[row[i], col[i], j] = np.uint8(rgb)


        # plt.imshow(self.state_img + self.maze*255, vmin=0, vmax=255)
        # plt.imshow(self.output_img)
        plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)
        plt.show(False)
        plt.pause(0.0001)

    def reset(self):
        return self._build_robot()

    def expert(self, robot_id=0, acs_seq=[]):

        if robot_id == self.robot_num-1 or self.agg_area<=self.goal_range:
            return self.expert_restart_session()

        # Replaning via breadth-first search
        if len(acs_seq)==0:
            BSF_Frontier = []
            while (self.loc[robot_id] == self.loc[robot_id+1]).all():
                robot_id += 1
            goal = np.copy(self.loc[robot_id+1,:])
            start = np.copy(self.loc[robot_id,:])
            cur = np.copy(start)
            # Initialize cost-to-go map
            costMap = np.copy(mazeData)
            BSF_Frontier.append(goal)
            cost = 100
            costMap[BSF_Frontier[0][0], BSF_Frontier[0][1]] = cost

            while len(BSF_Frontier) > 0 and (BSF_Frontier[0] != start).any():
                cost = costMap[BSF_Frontier[0][0], BSF_Frontier[0][1]] + 1
                for i in range(4):
                    new_pt = BSF_Frontier[0] + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                    if costMap[new_pt[0], new_pt[1]] == 1.0:
                        BSF_Frontier.append(new_pt)
                        costMap[new_pt[0], new_pt[1]] = cost
                BSF_Frontier.pop(0)
            costMap[costMap<100] = 0
            costMap[costMap>0] -= 99

            while (cur != goal).any():
                _cost_to_goal = costMap[cur[0], cur[1]] - 1
                for i in range(4):
                    new_pt = cur + np.array([np.cos(i * np.pi / 2), np.sin(i * np.pi / 2)]).astype(int)
                    if (costMap[new_pt[0], new_pt[1]] == _cost_to_goal):
                        if np.absolute(new_pt - cur)[0]:
                            action = (np.amax([0, (new_pt - cur)[0]]))
                        if np.absolute(new_pt - cur)[1]:
                            action = (np.amax([2, 2 + (new_pt - cur)[1]]))
                        cur = new_pt
                        acs_seq.append(action)
                        break

        action = acs_seq.pop(0)

        return robot_id, action, acs_seq

    def expert_restart_session(self):
        if self.agg_area <= self.goal_range:
            self.reset()
        robot_id = 0
        acs_seq = []
        return self.expert(robot_id, acs_seq)

def main(MazeEnv):
    env = MazeEnv()
    env.render()
    plt.pause(2)
    n_epochs = 10000
    steps  = 0
    rewards = 0.0
    acs_seq = []
    robot_id = -1
    import time
    start = time.time()
    for i in range(n_epochs):
        if i%50 ==0:
            now = time.time()
            print('step {} time elapse {}'.format(i, now-start))
            start = now
        steps+=1
        #next_action = np.random.randint(4,size = 1)
        robot_id, next_action, acs_seq = env.expert(robot_id, acs_seq)
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
    main(Maze1203AggEnv)


