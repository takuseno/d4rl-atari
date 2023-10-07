import numpy as np
import gym
import cv2

from gym import spaces
from gym.wrappers import AtariPreprocessing, TransformReward, FrameStack
from .offline_env import OfflineEnv


def capitalize_game_name(game):
    game = game.replace('-', '_')
    return ''.join([g.capitalize() for g in game.split('_')])


class AtariEnv(gym.Env):

    def __init__(self,
                 game,
                 stack=False,
                 sticky_action=False,
                 clip_reward=False,
                 terminal_on_life_loss=False,
                 **kwargs):
        # set action_probability=0.25 if sticky_action=True
        env_id = '{}NoFrameskip-v{}'.format(game, 0 if sticky_action else 4)

        # use official atari wrapper
        params = {}
        if "render_mode" in kwargs:
            params["render_mode"] = kwargs["render_mode"]
        env = AtariPreprocessing(gym.make(env_id, **params),
                                 terminal_on_life_loss=terminal_on_life_loss)

        if stack:
            env = FrameStack(env, num_stack=4)

        if clip_reward:
            env = TransformReward(env, lambda r: np.clip(r, -1.0, 1.0))

        self._env = env

        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def step(self, action):
        return self._env.step(action)

    def reset(self, *args, **kwargs):
        return self._env.reset(*args, **kwargs)

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    @property
    def render_mode(self):
        return self._env.render_mode


class OfflineAtariEnv(AtariEnv, OfflineEnv):

    def __init__(self, **kwargs):
        game = capitalize_game_name(kwargs['game'])
        del kwargs['game']
        AtariEnv.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)
