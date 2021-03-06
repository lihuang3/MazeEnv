from gym.envs.registration import registry, register, make, spec

# Swarm Path Planning
# ----------------------------------------
register(
    id='Maze0524Env-v5',
    entry_point='gym.envs.my_games:Maze0524Env5',
    max_episode_steps=400,
)

register(
    id='Maze0524Env-v4',
    entry_point='gym.envs.my_games:Maze0524Env4',
    max_episode_steps=360,
)


register(
    id='Maze0524Env-v3',
    entry_point='gym.envs.my_games:Maze0524Env3',
    max_episode_steps=360,
)


register(
    id='Maze0524Env-v2',
    entry_point='gym.envs.my_games:Maze0524Env2',
    max_episode_steps=300,
)


register(
    id='Maze0524Env-v1',
    entry_point='gym.envs.my_games:Maze0524Env1',
    max_episode_steps=480,
)


register(
    id='Maze0524Env-v0',
    entry_point='gym.envs.my_games:Maze0524Env',
    max_episode_steps=420,
)

register(
    id='Maze0523Env-v4',
    entry_point='gym.envs.my_games:Maze0523Env4',
    max_episode_steps=272,
)

register(
    id='Maze0523Env-v5',
    entry_point='gym.envs.my_games:Maze0523Env5',
    max_episode_steps=272,
)

register(
    id='Maze0523Env-v2',
    entry_point='gym.envs.my_games:Maze0523Env2',
    max_episode_steps=200,
)

register(
    id='Maze0523Env-v3',
    entry_point='gym.envs.my_games:Maze0523Env3',
    max_episode_steps=188,
)


register(
    id='Maze0523Env-v0',
    entry_point='gym.envs.my_games:Maze0523Env',
    max_episode_steps=280,
)

register(
    id='Maze0523Env-v1',
    entry_point='gym.envs.my_games:Maze0523Env1',
    max_episode_steps=260,
)

register(
    id='Maze0522Env-v5',
    entry_point='gym.envs.my_games:Maze0522Env5',
    max_episode_steps=280,
)

register(
    id='Maze0522Env-v4',
    entry_point='gym.envs.my_games:Maze0522Env4',
    max_episode_steps=280,
)

register(
    id='Maze0522Env-v3',
    entry_point='gym.envs.my_games:Maze0522Env3',
    max_episode_steps=220,
)

register(
    id='Maze0522Env-v2',
    entry_point='gym.envs.my_games:Maze0522Env2',
    max_episode_steps=220,
)


register(
    id='Maze0522Env-v0',
    entry_point='gym.envs.my_games:Maze0522Env',
    max_episode_steps=300,
)

register(
    id='Maze0522Env-v1',
    entry_point='gym.envs.my_games:Maze0522Env1',
    max_episode_steps=280,
)

register(
    id='Maze0521Env-v3',
    entry_point='gym.envs.my_games:Maze0521Env3',
    max_episode_steps=300,
)

register(
    id='Maze0521Env-v2',
    entry_point='gym.envs.my_games:Maze0521Env2',
    max_episode_steps=300,
)


register(
    id='Maze0521Env-v1',
    entry_point='gym.envs.my_games:Maze0521Env1',
    max_episode_steps=300,
)


register(
    id='Maze0521Env-v0',
    entry_point='gym.envs.my_games:Maze0521Env',
    max_episode_steps=300,
)


register(
    id='Maze0519Env-v3',
    entry_point='gym.envs.my_games:Maze0519Env3',
    max_episode_steps=1500,
)

register(
    id='Maze0519Env-v2',
    entry_point='gym.envs.my_games:Maze0519Env2',
    max_episode_steps=1500,
)

register(
    id='Maze0519Env-v0',
    entry_point='gym.envs.my_games:Maze0519Env',
    max_episode_steps=1500,
)

register(
    id='Maze0518Env-v0',
    entry_point='gym.envs.my_games:Maze0518Env',
    max_episode_steps=1500,
)

register(
    id='Maze1202Env-v0',
    entry_point='gym.envs.my_games:Maze1202Env',
    max_episode_steps=1000,
)

register(
    id='Maze1202Env-v1',
    entry_point='gym.envs.my_games:Maze1202Env1',
    max_episode_steps=1000,
)

register(
    id='Maze1204Env-v0',
    entry_point='gym.envs.my_games:Maze1204Env',
    max_episode_steps=2000,
)

register(
    id='Maze1204Env-v1',
    entry_point='gym.envs.my_games:Maze1204Env1',
    max_episode_steps=3000,
)

register(
    id='Maze1203Env-v5',
    entry_point='gym.envs.my_games:Maze1203Env5',
    max_episode_steps=1500,
)

register(
    id='Maze1203Env-v4',
    entry_point='gym.envs.my_games:Maze1203Env4',
    max_episode_steps=1500,
)

register(
    id='Maze1203Env-v3',
    entry_point='gym.envs.my_games:Maze1203Env3',
    max_episode_steps=2000,
)

register(
    id='Maze1203Env-v2',
    entry_point='gym.envs.my_games:Maze1203Env2',
    max_episode_steps=1500,
)

register(
    id='Maze1203Env-v1',
    entry_point='gym.envs.my_games:Maze1203Env1',
    max_episode_steps=2000,
)

register(
    id='Maze1203Env-v0',
    entry_point='gym.envs.my_games:Maze1203Env',
    max_episode_steps=1500,
)


register(
    id='Maze0319Env-v0',
    entry_point='gym.envs.my_games:Maze0319Env',
    max_episode_steps=1000,
)

register(
    id='Maze0319Env-v1',
    entry_point='gym.envs.my_games:Maze0319Env1',
    max_episode_steps=1000,
)

register(
    id='Maze0318Env-v0',
    entry_point='gym.envs.my_games:Maze0318Env',
    max_episode_steps=1000,
)

register(
    id='Maze0318Env-v1',
    entry_point='gym.envs.my_games:Maze0318Env1',
    max_episode_steps=1000,
)

register(
    id='Maze0122Env-v0',
    entry_point='gym.envs.my_games:Maze0122Env',
    max_episode_steps=4000,
)

register(
    id='Maze0122Env-v1',
    entry_point='gym.envs.my_games:Maze0122Env1',
    max_episode_steps=5000,
)

register(
    id='Maze0122Env-v2',
    entry_point='gym.envs.my_games:Maze0122Env2',
    max_episode_steps=2400,
)

register(
    id='Maze0122Env-v3',
    entry_point='gym.envs.my_games:Maze0122Env3',
    max_episode_steps=4000,
)

register(
    id='Maze0122Env-v4',
    entry_point='gym.envs.my_games:Maze0122Env4',
    max_episode_steps=4000,
)

register(
    id='Maze0110Env-v0',
    entry_point='gym.envs.my_games:Maze0110Env',
    max_episode_steps=3000,
)

register(
    id='Maze0110Env-v1',
    entry_point='gym.envs.my_games:Maze0110Env1',
    max_episode_steps=3000,
)

register(
    id='Maze0110Env-v2',
    entry_point='gym.envs.my_games:Maze0110Env2',
    max_episode_steps=3000,
)


register(
    id='Maze1218Env-v0',
    entry_point='gym.envs.my_games:Maze1218Env',
    max_episode_steps=2500,
)

register(
    id='Maze1217Env-v0',
    entry_point='gym.envs.my_games:Maze1217Env',
    max_episode_steps=2000,
)

register(
    id='Maze1203AggEnv-v0',
    entry_point='gym.envs.my_games:Maze1203AggEnv',
    max_episode_steps=2000,
)

register(
    id='FishWeir-v0',
    entry_point='gym.envs.my_games:FishWeirEnv',
    max_episode_steps=1000,
)


register(
    id='Maze1126Env-v0',
    entry_point='gym.envs.my_games:Maze1126Env',
    max_episode_steps=5000,
)

register(
    id='MazeEnv-v0',
    entry_point='gym.envs.my_games:MazeEnv',
    max_episode_steps=1000,
)

register(
    id='MazeEnv-v1',
    entry_point='gym.envs.my_games:MazeEnv1',
    max_episode_steps=1000,
)


register(
    id='MazeEnv-v2',
    entry_point='gym.envs.my_games:MazeEnv2',
    max_episode_steps=1500,
)

register(
    id='MazeEnv-v3',
    entry_point='gym.envs.my_games:MazeEnv3',
    max_episode_steps=1000,
)

register(
    id='MazeEnv-v4',
    entry_point='gym.envs.my_games:MazeEnv4',
    max_episode_steps=1000,
)

register(
    id='MazeEnvNOP-v0',
    entry_point='gym.envs.my_games:MazeEnvNOP0',
    max_episode_steps=1000,
)

register(
    id='MazeEnvNOP-v2',
    entry_point='gym.envs.my_games:MazeEnvNOP2',
    max_episode_steps=1000,
)

register(
    id='MazeEnvNOP-v10',
    entry_point='gym.envs.my_games:MazeEnvNOP10',
    max_episode_steps=3000,
)


register(
    id='MazeEnvAgg-v0',
    entry_point='gym.envs.my_games:MazeEnvAgg0',
    max_episode_steps=800,
)

register(
    id='MazeEnvAgg-v1',
    entry_point='gym.envs.my_games:MazeEnvAgg1',
    max_episode_steps=800,
)



register(
    id='MazeEnvRGB-v0',
    entry_point='gym.envs.my_games:MazeEnvRGB0',
    max_episode_steps=1000,
)

# Algorithmic
# ----------------------------------------
register(
    id='Copy-v0',
    entry_point='gym.envs.algorithmic:CopyEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='RepeatCopy-v0',
    entry_point='gym.envs.algorithmic:RepeatCopyEnv',
    max_episode_steps=200,
    reward_threshold=75.0,
)

register(
    id='ReversedAddition-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 2},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='ReversedAddition3-v0',
    entry_point='gym.envs.algorithmic:ReversedAdditionEnv',
    kwargs={'rows' : 3},
    max_episode_steps=200,
    reward_threshold=25.0,
)

register(
    id='DuplicatedInput-v0',
    entry_point='gym.envs.algorithmic:DuplicatedInputEnv',
    max_episode_steps=200,
    reward_threshold=9.0,
)

register(
    id='Reverse-v0',
    entry_point='gym.envs.algorithmic:ReverseEnv',
    max_episode_steps=200,
    reward_threshold=25.0,
)

# Classic
# ----------------------------------------

register(
    id='CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200,
    reward_threshold=195.0,
)

register(
    id='CartPole-v1',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=475.0,
)

register(
    id='MountainCar-v0',
    entry_point='gym.envs.classic_control:MountainCarEnv',
    max_episode_steps=200,
    reward_threshold=-110.0,
)

register(
    id='MountainCarContinuous-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv',
    max_episode_steps=999,
    reward_threshold=90.0,
)

register(
    id='Pendulum-v0',
    entry_point='gym.envs.classic_control:PendulumEnv',
    max_episode_steps=200,
)

register(
    id='Acrobot-v1',
    entry_point='gym.envs.classic_control:AcrobotEnv',
    max_episode_steps=500,
)

# Box2d
# ----------------------------------------

register(
    id='LunarLander-v2',
    entry_point='gym.envs.box2d:LunarLander',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='LunarLanderContinuous-v2',
    entry_point='gym.envs.box2d:LunarLanderContinuous',
    max_episode_steps=1000,
    reward_threshold=200,
)

register(
    id='BipedalWalker-v2',
    entry_point='gym.envs.box2d:BipedalWalker',
    max_episode_steps=1600,
    reward_threshold=300,
)

register(
    id='BipedalWalkerHardcore-v2',
    entry_point='gym.envs.box2d:BipedalWalkerHardcore',
    max_episode_steps=2000,
    reward_threshold=300,
)

register(
    id='CarRacing-v0',
    entry_point='gym.envs.box2d:CarRacing',
    max_episode_steps=1000,
    reward_threshold=900,
)

# Toy Text
# ----------------------------------------

register(
    id='Blackjack-v0',
    entry_point='gym.envs.toy_text:BlackjackEnv',
)

register(
    id='KellyCoinflip-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipEnv',
    reward_threshold=246.61,
)
register(
    id='KellyCoinflipGeneralized-v0',
    entry_point='gym.envs.toy_text:KellyCoinflipGeneralizedEnv',
)

register(
    id='FrozenLake-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4'},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

register(
    id='FrozenLake8x8-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '8x8'},
    max_episode_steps=200,
    reward_threshold=0.99, # optimum = 1
)

register(
    id='CliffWalking-v0',
    entry_point='gym.envs.toy_text:CliffWalkingEnv',
)

register(
    id='NChain-v0',
    entry_point='gym.envs.toy_text:NChainEnv',
    max_episode_steps=1000,
)

register(
    id='Roulette-v0',
    entry_point='gym.envs.toy_text:RouletteEnv',
    max_episode_steps=100,
)

register(
    id='Taxi-v2',
    entry_point='gym.envs.toy_text.taxi:TaxiEnv',
    reward_threshold=8, # optimum = 8.46
    max_episode_steps=200,
)

register(
    id='GuessingGame-v0',
    entry_point='gym.envs.toy_text.guessing_game:GuessingGame',
    max_episode_steps=200,
)

register(
    id='HotterColder-v0',
    entry_point='gym.envs.toy_text.hotter_colder:HotterColder',
    max_episode_steps=200,
)

# Mujoco
# ----------------------------------------

# 2D

register(
    id='Reacher-v2',
    entry_point='gym.envs.mujoco:ReacherEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)

register(
    id='Pusher-v2',
    entry_point='gym.envs.mujoco:PusherEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Thrower-v2',
    entry_point='gym.envs.mujoco:ThrowerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='Striker-v2',
    entry_point='gym.envs.mujoco:StrikerEnv',
    max_episode_steps=100,
    reward_threshold=0.0,
)

register(
    id='InvertedPendulum-v2',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
    max_episode_steps=1000,
    reward_threshold=950.0,
)

register(
    id='InvertedDoublePendulum-v2',
    entry_point='gym.envs.mujoco:InvertedDoublePendulumEnv',
    max_episode_steps=1000,
    reward_threshold=9100.0,
)

register(
    id='HalfCheetah-v2',
    entry_point='gym.envs.mujoco:HalfCheetahEnv',
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

register(
    id='Hopper-v2',
    entry_point='gym.envs.mujoco:HopperEnv',
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

register(
    id='Swimmer-v2',
    entry_point='gym.envs.mujoco:SwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)

register(
    id='Walker2d-v2',
    max_episode_steps=1000,
    entry_point='gym.envs.mujoco:Walker2dEnv',
)

register(
    id='Ant-v2',
    entry_point='gym.envs.mujoco:AntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='Humanoid-v2',
    entry_point='gym.envs.mujoco:HumanoidEnv',
    max_episode_steps=1000,
)

register(
    id='HumanoidStandup-v2',
    entry_point='gym.envs.mujoco:HumanoidStandupEnv',
    max_episode_steps=1000,
)

# Robotics
# ----------------------------------------

def _merge(a, b):
    a.update(b)
    return a

for reward_type in ['sparse', 'dense']:
    suffix = 'Dense' if reward_type == 'dense' else ''
    kwargs = {
        'reward_type': reward_type,
    }

    # Fetch
    register(
        id='FetchSlide{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchSlideEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPickAndPlace{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchPickAndPlaceEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchReach{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='FetchPush{}-v1'.format(suffix),
        entry_point='gym.envs.robotics:FetchPushEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    # Hand
    register(
        id='HandReach{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandReachEnv',
        kwargs=kwargs,
        max_episode_steps=50,
    )

    register(
        id='HandManipulateBlockRotateZ{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'z'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateParallel{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'parallel'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockRotateXYZ{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateBlockFull{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateBlock{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandBlockEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggRotate{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulateEggFull{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulateEgg{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandEggEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenRotate{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'ignore', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    register(
        id='HandManipulatePenFull{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

    # Alias for "Full"
    register(
        id='HandManipulatePen{}-v0'.format(suffix),
        entry_point='gym.envs.robotics:HandPenEnv',
        kwargs=_merge({'target_position': 'random', 'target_rotation': 'xyz'}, kwargs),
        max_episode_steps=100,
    )

# Atari
# ----------------------------------------

# # print ', '.join(["'{}'".format(name.split('.')[0]) for name in atari_py.list_games()])
for game in ['air_raid', 'alien', 'amidar', 'assault', 'asterix', 'asteroids', 'atlantis',
    'bank_heist', 'battle_zone', 'beam_rider', 'berzerk', 'bowling', 'boxing', 'breakout', 'carnival',
    'centipede', 'chopper_command', 'crazy_climber', 'demon_attack', 'double_dunk',
    'elevator_action', 'enduro', 'fishing_derby', 'freeway', 'frostbite', 'gopher', 'gravitar',
    'hero', 'ice_hockey', 'jamesbond', 'journey_escape', 'kangaroo', 'krull', 'kung_fu_master',
    'montezuma_revenge', 'ms_pacman', 'name_this_game', 'phoenix', 'pitfall', 'pong', 'pooyan',
    'private_eye', 'qbert', 'riverraid', 'road_runner', 'robotank', 'seaquest', 'skiing',
    'solaris', 'space_invaders', 'star_gunner', 'tennis', 'time_pilot', 'tutankham', 'up_n_down',
    'venture', 'video_pinball', 'wizard_of_wor', 'yars_revenge', 'zaxxon']:
    for obs_type in ['image', 'ram']:
        # space_invaders should yield SpaceInvaders-v0 and SpaceInvaders-ram-v0
        name = ''.join([g.capitalize() for g in game.split('_')])
        if obs_type == 'ram':
            name = '{}-ram'.format(name)

        nondeterministic = False
        if game == 'elevator_action' and obs_type == 'ram':
            # ElevatorAction-ram-v0 seems to yield slightly
            # non-deterministic observations about 10% of the time. We
            # should track this down eventually, but for now we just
            # mark it as nondeterministic.
            nondeterministic = True

        register(
            id='{}-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'repeat_action_probability': 0.25},
            max_episode_steps=10000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        # Standard Deterministic (as in the original DeepMind paper)
        if game == 'space_invaders':
            frameskip = 3
        else:
            frameskip = 4

        # Use a deterministic frame skip.
        register(
            id='{}Deterministic-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip, 'repeat_action_probability': 0.25},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}Deterministic-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': frameskip},
            max_episode_steps=100000,
            nondeterministic=nondeterministic,
        )

        register(
            id='{}NoFrameskip-v0'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1, 'repeat_action_probability': 0.25}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )

        # No frameskip. (Atari has no entropy source, so these are
        # deterministic environments.)
        register(
            id='{}NoFrameskip-v4'.format(name),
            entry_point='gym.envs.atari:AtariEnv',
            kwargs={'game': game, 'obs_type': obs_type, 'frameskip': 1}, # A frameskip of 1 means we get every frame
            max_episode_steps=frameskip * 100000,
            nondeterministic=nondeterministic,
        )


# Unit test
# ---------

register(
    id='CubeCrash-v0',
    entry_point='gym.envs.unittest:CubeCrash',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashSparse-v0',
    entry_point='gym.envs.unittest:CubeCrashSparse',
    reward_threshold=0.9,
    )
register(
    id='CubeCrashScreenBecomesBlack-v0',
    entry_point='gym.envs.unittest:CubeCrashScreenBecomesBlack',
    reward_threshold=0.9,
    )

register(
    id='MemorizeDigits-v0',
    entry_point='gym.envs.unittest:MemorizeDigits',
    reward_threshold=20,
    )

