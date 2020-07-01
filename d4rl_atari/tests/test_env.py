import gym
import d4rl_atari
import os


def test_env():
    env = gym.make('breakout-mixed-v0')

    observation = env.reset()
    assert observation.shape == (84, 84, 1)

    observation = env.step(1)[0]
    assert observation.shape == (84, 84, 1)

    # skip this on CI due to memory allocation error
    if not os.getenv('CI'):
        dataset = env.get_dataset()
        assert dataset['observations'].shape == (1000000, 84, 84)
        assert dataset['actions'].shape == (1000000, )
        assert dataset['rewards'].shape == (1000000, )
        assert dataset['terminals'].shape == (1000000, )
