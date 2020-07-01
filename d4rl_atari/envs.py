import numpy as np
import gym
import cv2

from gym import spaces
from .offline_env import OfflineEnv


def capitalize_game_name(game):
    game = game.replace('-', '_')
    return ''.join([g.capitalize() for g in game.split('_')])


class AtariEnv(gym.Env):
    def __init__(self, game, frameskip=4, **kwargs):
        # set action_probability=0.25
        env_id = '{}NoFrameskip-v0'.format(game)
        atari_env = gym.make(env_id)
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(84, 84, 1),
                                            dtype=np.uint8)
        self.action_space = atari_env.action_space
        self.env = atari_env.env

        self.frameskip = frameskip
        self.screen_buffer = np.zeros((2, 84, 84), dtype=np.uint8)

    def reset(self):
        self.env.reset()
        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        return self._pool_and_resize()

    def step(self, action):
        accumulated_reward = 0.0
        for time_step in range(self.frameskip):
            _, reward, done, info = self.env.step(action)

            accumulated_reward += reward

            if done:
                break
            elif time_step >= self.frameskip - 2:
                t = time_step - (self.frameskip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        observation = self._pool_and_resize()

        return observation, accumulated_reward, done, info

    def _fetch_grayscale_observation(self, output):
        self.env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        np.maximum(self.screen_buffer[0],
                   self.screen_buffer[1],
                   out=self.screen_buffer[0])

        resized_screen = cv2.resize(self.screen_buffer[0], (84, 84),
                                    interpolation=cv2.INTER_AREA)

        image = np.asarray(resized_screen, dtype=np.uint8)

        return np.expand_dims(image, axis=2)


class OfflineAtariEnv(AtariEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = capitalize_game_name(kwargs['game'])
        del kwargs['game']
        AtariEnv.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)
