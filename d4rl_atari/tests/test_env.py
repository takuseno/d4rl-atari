import gym
import d4rl_atari


def test_env():
    env = gym.make('breakout-mixed-v0')

    dataset = env.get_dataset()

    assert dataset['observations'].shape == (1000000, 84, 84)
    assert dataset['actions'].shape == (1000000, )
    assert dataset['rewards'].shape == (1000000, )
    assert dataset['terminals'].shape == (1000000, )

    observation = env.reset()
    assert observation.shape == (84, 84, 1)

    #observation = env.step(1)
    #assert observation.shape[0] == (84, 84, 1)
