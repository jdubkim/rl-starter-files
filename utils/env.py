import gym
import gym_minigrid

import numpy as np


class ResizeObservation(gym.ObservationWrapper):
    r"""Downsample the image observation to a square image."""

    def __init__(self, env, size):
        super().__init__(env)
        self._size = size 
        self._keys = [
            k for k, v in env.observation_space.spaces.items() if len(v.shape) > 1 and v.shape[:2] != size]
        
        if self._keys:
            from PIL import Image
            self._Image = Image

        self.observation_space = self.env.observation_space
        for key in self._keys:
            obs_shape = (size[0], size[1], 3)
            self.observation_space.spaces[key] = \
                gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        for key in self._keys:
            observation[key] = self._resize(observation[key])
        return observation

    def _resize(self, image):
        image = self._Image.fromarray(image)
        iamge = image.resize(self._size, self._Image.NEAREST)
        image = np.array(image)
        return image


def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key)
    env.seed(seed)
    if 'MiniGrid' in env_key:
        env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(env)
        env = ResizeObservation(env, (64, 64))
    
    return env
