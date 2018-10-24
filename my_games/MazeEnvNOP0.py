"""
  gym id: MazeEnvRGB-v0
  Input: RGB images
  Reward: region range basis (easy)
  Training: from scratch
"""

import numpy as np, random, sys, time, os
import gym
from gym import error, spaces, utils, core
from gym.utils import seeding
import matplotlib.pyplot as plt

plt.ion()

from time import sleep

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from MazeEnv import *


class MazeEnvNOP0(MazeEnv):
  def __init__(self):
    global mazeData, costData, centerline, freespace, mazeHeight, mazeWidth, robot_marker
    dir_path = os.path.dirname(os.path.realpath(__file__))

    self.map_data_dir = dir_path + '/MapData'
    self.satur = 0.02
    robot_marker = 150
    self.goal_range = 10
    self.agg_rate = 0.0
    self.actions = [1, 2, 3, 4]  # {up, down, left ,right}
    self.action_dict = {0: (-1, 0), 1: (1, 0), 2: (-1, 1), 3: (1, 1)}  # {up, down, left ,right}
    self.n_actions = len(self.actions)
    self.action_space = spaces.Discrete(self.n_actions)
    mazeData, costData, centerline, freespace = self._load_data(self.map_data_dir)
    mazeHeight, mazeWidth = mazeData.shape
    self.observation_space = spaces.Box(low=0, high=255, shape=(mazeHeight, mazeWidth, 1), dtype=np.uint8)
    self.seed()
    self.maze = np.ones((mazeHeight, mazeWidth)) - mazeData
    self.centerline = np.ones((mazeHeight, mazeWidth)) - centerline
    self.freespace = np.ones((mazeHeight, mazeWidth)) - freespace
    # self.goal = np.array([73, 10])
    self.goal = np.array([31, 52])
    self.init_state = []
    self.reset()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _load_data(self, data_directory):
    mazeData = np.loadtxt(data_directory + '/scaled_maze7.csv').astype(int)
    freespace = np.loadtxt(data_directory + '/scaled_maze7_freespace.csv').astype(int)
    costData = np.loadtxt(data_directory + '/scaled_maze7_costmap1.csv').astype(int)
    centerline = np.loadtxt(data_directory + '/scaled_maze7_centerline.csv').astype(int)
    return mazeData, costData, centerline, freespace

  def _build_robot(self):
    row, col = np.nonzero(freespace)
    self.reward_grad = np.zeros(7).astype(np.uint8)
    # try:
    #     self.init_state
    #     self.init_state_img
    #     self.state = self.init_state
    #     self.state_img = self.init_state_img
    # except AttributeError:
    self.robot_num = 100  # len(row)
    self.robot = random.sample(range(row.shape[0]), self.robot_num)
    self.state = np.zeros(np.shape(mazeData)).astype(int)
    self.state_img = np.copy(self.state)
    for i in range(self.robot_num):
      self.state[row[self.robot[i]], col[self.robot[i]]] += robot_marker
      self.state_img[row[self.robot[i]] - 1:row[self.robot[i]] + 2,
      col[self.robot[i]] - 1:col[self.robot[i]] + 2] = robot_marker * np.ones([3, 3])
    self.state_img = self.state
    self.init_state = self.state
    self.init_state_img = self.state_img

    self.output_img = self.state_img + self.maze * 255

    # return self.output_img

    return (np.expand_dims(self.output_img, axis=2))

  def step(self, action):

    next_direction, next_axis = self.action_dict[action]

    next_state = np.roll(self.state, next_direction, axis=next_axis)

    # Collision check
    collision = np.logical_and(next_state, self.freespace) * next_state
    next_state *= np.logical_xor(next_state, self.freespace)

    # Move robots in the obstacle area back to previous grids and obtain the next state
    ## Case 1: overlapping with population index
    ######next_state += np.roll(collision, -next_direction, axis=next_axis)
    ## Case 2: overlapping w/o population index (0: no robot; 1: robot(s) exits)
    # next_state = np.logical_or(np.roll(collision, -next_direction, axis=next_axis), next_state).astype(int)
    # Case 3: non-overlapping
    new_collision = collision


    while(True):
      collision = np.roll(new_collision, -next_direction, axis=next_axis)
      next_state += collision

      new_collision = 0*np.copy(collision)
      redun = np.where((next_state > self.satur * self.robot_num * robot_marker) & (collision>0))

      new_collision[redun] = next_state[redun]-int(self.satur*self.robot_num*robot_marker)
      next_state[redun] = int(self.satur * self.robot_num * robot_marker)

      if( np.squeeze(np.shape( np.where(new_collision>0)[0])) == 0):
        break

    # next_state *= robot_marker   # Mark robot with intensity 150

    row, col = np.nonzero(next_state)

    self.state_img *= 0  # np.zeros([mazeHeight,mazeWidth])

    for i in range(row.shape[0]):
      self.state_img[row[i] - 1:row[i] + 2, col[i] - 1:col[i] + 2] = robot_marker * np.ones([3, 3])


    self.state = next_state

    self.output_img = (self.state_img + self.maze * 255).astype(np.uint8)

    self.state_cost_matrix = self.state * costData / robot_marker

    self.agg_rate = np.sum( self.state[np.array(costData)<=self.goal_range]  / robot_marker) / self.robot_num

    cost_to_go = np.sum(self.state_cost_matrix)

    done = False
    reward = -.1

    if cost_to_go <= self.goal_range * self.robot_num:
      done = True
      reward = 100.0
    elif cost_to_go <= 2 * self.goal_range * self.robot_num and not self.reward_grad[0]:
      self.reward_grad[0] = 1
      reward = 20.0
    elif cost_to_go <= 3 * self.goal_range and not self.reward_grad[1]:
      self.reward_grad[1] = 1
      reward = 5.0

    info = {}

    return (self.output_img, reward, done, info)

  def render(self, mode='human'):
    # print("")
    row, col = np.nonzero(self.state)
    render_image = np.copy(self.state).astype(np.int16)
    for i in range(row.shape[0]):
     render_image[row[i] - 1:row[i] + 2, col[i] - 1:col[i] + 2] += self.state[row[i], col[i]] * np.ones([3, 3]).astype(
       np.int16)

    row, col = np.nonzero(render_image)
    min_robots = robot_marker
    max_robots = self.robot_num*robot_marker*self.satur

    tmp = np.copy(self.maze)
    tmp[tmp < 1] = 255

    rgb_render_image = np.stack((render_image + tmp,) * 3, -1)

    for i in range(row.shape[0]):
      value = render_image[row[i], col[i]]
      ratio = 2. * (value - min_robots) / (max_robots - min_robots)
      b = np.uint8(min(max(0, 255 * (1 - ratio)), 255))
      r = np.uint8(min(max(0, 255 * (ratio - 1)), 255))
      g = np.uint8(255 - b - r)

      for j, rgb in enumerate([r, g, b]):
        rgb_render_image[row[i], col[i], j] = np.uint8(rgb)

    #plt.imshow(self.output_img)
    #plt.imshow(render_image.astype(np.uint8), vmin=0, vmax=255)
    plt.imshow(rgb_render_image.astype(np.uint8), vmin=0, vmax=255)

    plt.text(35, 5, "Agg. rate %.1f"%(100.*self.agg_rate)+"%", fontsize = 12, color='white')
    plt.show(False)
    plt.pause(0.0001)
    plt.gcf().clear()

  def reset(self):
    return self._build_robot()

  def expert(self, robot_loc):

    _cost_to_goal = np.sum(self.state * costData / robot_marker)
    if not len(robot_loc) or _cost_to_goal <= self.robot_num * self.goal_range:
      return self.expert_restart_session()

    _cost_to_goal = costData[robot_loc[0], robot_loc[1]]
    if _cost_to_goal > 1:
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
        elif (costData[new_pt[0], new_pt[1]] == _cost_to_goal + 1):
          if np.absolute(new_pt - robot_loc)[0]:
            action = (np.amax([0, (new_pt - robot_loc)[0]]))
          if np.absolute(new_pt - robot_loc)[1]:
            action = (np.amax([2, 2 + (new_pt - robot_loc)[1]]))

          robot_loc = new_pt
      return action, robot_loc


    else:
      return self.expert_restart_session()

  def expert_restart_session(self):
    if (np.sum(self.state * costData / robot_marker) <= self.robot_num * self.goal_range):
      self.reset()
    robot_loc = np.unravel_index(np.argmax((self.state > 0) * costData), self.state.shape)
    return self.expert(robot_loc)


def main(MazeEnv):
  env = MazeEnv()
  env.render()
  plt.pause(2)

  n_epochs = 10000
  robot_loc = []
  steps = 0
  rewards = 0.0
  for i in range(n_epochs):
    steps += 1
    # next_action = np.random.randint(4,size = 1)
    next_action, robot_loc = env.expert(robot_loc)
    state_img, reward, done, _ = env.step(next_action)
    rewards += reward
    env.render()
    print('Step = %d, rewards = %.1f, reward = %.1f, done = %d' % (steps, rewards, reward, done), end='\r')
    if done:
      print('\n')

    if done:
      steps = 0
      rewards = 0.0
      plt.pause(2)
      env.reset()


if __name__ == '__main__':
  main(MazeEnvNOP0)


