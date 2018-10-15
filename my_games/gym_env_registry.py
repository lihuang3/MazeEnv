import gym

from collections import defaultdict
_game_envs = defaultdict(set)
for env in gym.envs.registry.all():
    # TODO: solve this with regexes
    env_type = env._entry_point.split(':')[0].split('.')[-1]
    _game_envs[env_type].add(env.id)

# reading benchmark names directly from retro requires
# importing retro here, and for some reason that crashes tensorflow
# in ubuntu
_game_envs['retro'] = {
    'BubbleBobble-Nes',
    'SuperMarioBros-Nes',
    'TwinBee3PokoPokoDaimaou-Nes',
    'SpaceHarrier-Nes',
    'SonicTheHedgehog-Genesis',
    'Vectorman-Genesis',
    'FinalFight-Snes',
    'SpaceInvaders-Snes',
}


def get_env_type(env_id):
  if env_id in _game_envs.keys():
    env_type = env_id
    env_id = [g for g in _game_envs[env_type]][0]
  else:
    env_type = None
    for g, e in _game_envs.items():
      if env_id in e:
        env_type = g
        break
    assert env_type is not None, 'env_id {} is not recognized in env types'.format(env_id, _game_envs.keys())

  return env_type, env_id



env_type, env_id = get_env_type('MazeEnv-v0')
print('env_type: {}'.format(env_type))

