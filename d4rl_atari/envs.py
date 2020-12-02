import numpy as np
import gym
import cv2

from gym import spaces
from .offline_env import OfflineEnv


def capitalize_game_name(game):
    game = game.replace('-', '_')
    return ''.join([g.capitalize() for g in game.split('_')])


# dopamine style Atari wrapper
class AtariEnv(gym.Env):
    def __init__(self,
                 game,
                 frameskip=4,
                 stack=True,
                 init_noop_steps=30,
                 clip_reward=False,
                 terminate_on_life_loss=False,
                 max_frames=108000,
                 **kwargs):
        # set action_probability=0.25
        env_id = '{}NoFrameskip-v0'.format(game)
        atari_env = gym.make(env_id)
        n_channels = 4 if stack else 1
        self.observation_space = spaces.Box(low=0,
                                            high=255,
                                            shape=(n_channels, 84, 84),
                                            dtype=np.uint8)
        self.action_space = atari_env.action_space
        self.env = atari_env.env
        self.screen_shape = atari_env.observation_space.shape[:2]
        self.stack = stack
        self.init_noop_steps = init_noop_steps
        self.clip_reward = clip_reward
        self.terminate_on_life_loss = terminate_on_life_loss
        self.max_frames = max_frames
        self.frameskip = frameskip

        self.screen_buffer = np.zeros((2, ) + self.screen_shape,
                                      dtype=np.uint8)
        self.stack_buffer = np.zeros((4, 84, 84), dtype=np.uint8)
        self.lives = 0
        self.episode_step = 0
        self.real_done = False

    def reset(self):
        if self.real_done:
            self.env.reset()

        # random initialization
        for _ in range(np.random.randint(self.init_noop_steps)):
            _, _, done, _ = self.env.step(0)
            if done:
                self.env.reset()

        self.lives = self.env.ale.lives()
        self.episode_step = 0
        self.real_done = False

        self._fetch_grayscale_observation(self.screen_buffer[0])
        self.screen_buffer[1].fill(0)
        self.stack_buffer.fill(0)
        observation = self._pool_and_resize()

        if self.stack:
            # fill blacks
            for _ in range(3):
                self._stack(observation)
            return self._stack(observation)

        return observation

    def step(self, action):
        accumulated_reward = 0.0
        for time_step in range(self.frameskip):
            _, reward, done, info = self.env.step(action)

            accumulated_reward += reward
            self.episode_step += 1
            self.real_done = done

            if self.terminate_on_life_loss:
                done = done or self._check_life_loss()

            if done:
                break

            if time_step >= self.frameskip - 2:
                t = time_step - (self.frameskip - 2)
                self._fetch_grayscale_observation(self.screen_buffer[t])

        observation = self._pool_and_resize()

        if self.stack:
            observation = self._stack(observation)

        if self.clip_reward:
            accumulated_reward = np.clip(accumulated_reward, -1, 1)

        if self.episode_step > self.max_frames:
            done = True
            self.real_done = True

        return observation, accumulated_reward, done, info

    def _check_life_loss(self):
        curr_lives = self.env.ale.lives()
        is_terminal = curr_lives < self.lives
        self.lives = curr_lives
        return is_terminal

    def _fetch_grayscale_observation(self, output):
        self.env.ale.getScreenGrayscale(output)
        return output

    def _pool_and_resize(self):
        max_pixel = np.max(self.screen_buffer, axis=0)
        self.screen_buffer[0][...] = max_pixel

        resized_screen = cv2.resize(self.screen_buffer[0], (84, 84),
                                    interpolation=cv2.INTER_AREA)

        image = np.asarray(resized_screen, dtype=np.uint8)

        return np.expand_dims(image, axis=0)

    def _stack(self, observation):
        self.stack_buffer = np.roll(self.stack_buffer, -1, axis=0)
        self.stack_buffer[-1][...] = observation[0]
        return self.stack_buffer

    def render(self, mode='human'):
        self.env.render(mode)


class OfflineAtariEnv(AtariEnv, OfflineEnv):
    def __init__(self, **kwargs):
        game = capitalize_game_name(kwargs['game'])
        del kwargs['game']
        AtariEnv.__init__(self, game=game, **kwargs)
        OfflineEnv.__init__(self, game=game, **kwargs)
