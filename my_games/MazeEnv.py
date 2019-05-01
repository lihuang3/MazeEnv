"""
  gym id: MazeEnv-v0
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

class MazeEnv(core.Env):
    def __init__(self):
        global mazeData, costData, centerline, freespace, mazeHeight, mazeWidth, robot_marker
        dir_path = os.path.dirname(os.path.realpath(__file__))

        self.map_data_dir = dir_path+'/MapData'
        self.save_costlog = False
        self.avgcost = np.empty((0))
        self.maxcost = np.empty((0))

        self.save_frame = False
        self.loc_hist = [[],[],[],[]]
        self.internal_steps = 0

        robot_marker = 150
        self.goal_range = 15
        self.actions = [1, 2, 3, 4] # {up, down, left ,right}
        self.action_map = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}  # {up, down, left ,right}
        self.n_actions = len(self.actions)
        self.action_space = spaces.Discrete(self.n_actions)
        mazeData, costData, freespace = self._load_data(self.map_data_dir)
        mazeHeight, mazeWidth = mazeData.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
        self.seed()
        self.maze = np.ones((mazeHeight, mazeWidth))-mazeData
        self.freespace = np.ones((mazeHeight, mazeWidth))-freespace
        self.goal = np.array([73, 10])
        # self.goal = np.array([31, 52])
        self.init_state = []
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _load_data(self, data_directory):
        filename = 'scaled_maze7'
        RGBmap = 'map1'
        self.raw_img = plt.imread(data_directory + '/'+RGBmap+'.png')
        mazeData = np.loadtxt(data_directory + '/'+filename+'.csv').astype(int)
        freespace = np.loadtxt(data_directory + '/'+filename+'_freespace.csv').astype(int)
        costData = np.loadtxt(data_directory + '/' +filename+ '_costmap.csv').astype(int)
        return mazeData, costData, freespace

    def _build_robot(self):
        self.internal_steps = 0

        if self.save_costlog:
            eplen = np.shape(self.avgcost)[0]
            if eplen > 100 and eplen < 500 :
                os.makedirs( os.path.join(ROOT_DIR, 'data'), exist_ok=True)
                filename = 'costplot.txt'
                output = np.append(self.avgcost, self.maxcost)
                np.savetxt(os.path.join(ROOT_DIR,'data', filename), output, fmt= '%.1f')
                exit()

        self.avgcost = np.empty((0))
        self.maxcost = np.empty((0)) 

        row, col = np.nonzero(freespace)
        self.reward_grad = np.zeros(40).astype(np.uint8)
        self.robot_num = 128 #len(row)
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

    def step(self,action):
        info = {}

        self.action = action
        # For sticky action rendering usage
        self.loc_hist[self.internal_steps] = self.loc
        self.internal_steps += 1
        self.internal_steps %= 4

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
        cost_to_go = np.sum(costData[self.loc[:, 0], self.loc[:, 1]])
        max_cost_agent = np.max(costData[self.loc[:, 0], self.loc[:, 1]])
        self.avgcost = np.append(self.avgcost, cost_to_go/float(self.robot_num))
        self.maxcost = np.append(self.maxcost, max_cost_agent)
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
            reward = 4
        elif cost_to_go <= 2*self.goal_range * self.robot_num and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward = 4
        elif cost_to_go <= 3*self.goal_range * self.robot_num  and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward = 4
        elif cost_to_go <= 4*self.goal_range * self.robot_num  and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward = 2
        elif cost_to_go <= 6*self.goal_range * self.robot_num  and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward = 2


        return done, reward

    def get_reward1(self):
        cost_to_go = np.sum(costData[self.loc[:, 0], self.loc[:, 1]])
        max_cost_agent = np.max(costData[self.loc[:, 0], self.loc[:, 1]])

        done = False
        reward = -.1

        if cost_to_go <= self.goal_range * self.robot_num:
            reward = 100
            done = True
        elif cost_to_go <= 2*self.goal_range * self.robot_num and not self.reward_grad[20]:
            self.reward_grad[20] = 1
            reward = 8
        elif cost_to_go <= 3*self.goal_range * self.robot_num  and not self.reward_grad[21]:
            self.reward_grad[21] = 1
            reward = 8
        elif cost_to_go <= 4*self.goal_range * self.robot_num  and not self.reward_grad[22]:
            self.reward_grad[22] = 1
            reward = 4
        elif cost_to_go <= 5*self.goal_range * self.robot_num  and not self.reward_grad[23]:
            self.reward_grad[23] = 1
            reward = 4
        elif cost_to_go <= 6*self.goal_range * self.robot_num  and not self.reward_grad[24]:
            self.reward_grad[24] = 1
            reward = 4
        elif cost_to_go <= 7*self.goal_range * self.robot_num  and not self.reward_grad[25]:
            self.reward_grad[25] = 1
            reward = 2
        elif cost_to_go <= 8*self.goal_range * self.robot_num  and not self.reward_grad[26]:
            self.reward_grad[26] = 1
            reward = 2


        return done, reward

    def render_config(self, loc, dot_size=20):
        # Load high-resolution
        plt.imshow(self.raw_img)
        h, w, _ = self.raw_img.shape
        h1, w1 = self.maze.shape
        # Draw goal range
        circle = plt.Circle((float(w)/w1*self.goal[1], float(h)/h1*self.goal[0]), 45, color='red', linewidth=3, fill=False)
        plt.gcf().gca().add_artist(circle)
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


    def _render(self, mode = 'human'):
        # plt.imshow(self.state_img + self.maze*255, vmin=0, vmax=255)
        plt.imshow(self.output_img)
        plt.show(False)
        plt.pause(0.0001)
        plt.gcf().clear()

    def render(self, mode = 'human'):
        idx = self.internal_steps - 5
        for i in range(4):
            idx = (idx + 1) % 4
            loc = self.loc_hist[i]
            plt.gcf().clear()

            self.render_config(loc, dot_size=40)
            if self.save_frame:
                fig = plt.gcf()
                fig.set_size_inches(3, 3)
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
    main(MazeEnv)


