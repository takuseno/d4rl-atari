import numpy as np
import gym
import d4rl_atari
import os

from d4rl_atari.offline_env import _stack


def test_stack():
    observations = np.random.random((1000, 84, 84))
    terminals = np.random.randint(2, size=1000)
    terminals[:100] = 0

    rets = _stack(observations, terminals)

    for ret in rets:
        assert ret.shape == (4, 84, 84)

    # check pointer
    assert np.all(observations[3] == rets[3][-1])
    rets[3][-1].fill(0)
    assert np.all(observations[3] == 0)


def test_env():
    env = gym.make('breakout-mixed-v0')

    observation = env.reset()
    assert observation.shape == (84, 84, 1)

    observation = env.step(1)[0]
    assert observation.shape == (84, 84, 1)

    # skip this on CI due to memory allocation error
    if not os.getenv('CI'):
        dataset = env.get_dataset()
        assert isinstance(dataset['observations'], list)
        assert len(dataset['observations']) == 1000000
        assert dataset['observations'][0].shape == (4, 84, 84)
        assert dataset['actions'].shape == (1000000, )
        assert dataset['rewards'].shape == (1000000, )
        assert dataset['terminals'].shape == (1000000, )
