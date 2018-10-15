
import numpy as np,matplotlib.pyplot as plt, sys
import gym

def main():
  env = gym.make('MazeEnv-v1')
  # env = gym.make('Breakout-v0')

  env.reset()
  env.render()
  plt.pause(2)
  n_epochs = 1000
  robot_loc = []

  for i in range(n_epochs):
    next_action = int(np.squeeze(np.random.randint(4,size = 1)) )
    # next_action, robot_loc = env.expert(robot_loc)
    state_img, reward, done, info = env.step(next_action)

    print('Step = {}, reward = {}, done = {}'.format(i, reward, done), end='\r')
    sys.stdout.flush()

    maybeepinfo = info.get('episode')
    if maybeepinfo:
      print('info %s'%(str(maybeepinfo)))
    env.render()
    if done:
      print('\n')
      env.reset()
      plt.pause(2)


if __name__ == '__main__':
  main()
