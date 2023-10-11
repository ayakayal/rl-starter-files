import gymnasium as gym
from minigrid.wrappers import RGBImgObsWrapper
from PIL import Image


def make_env(env_key, seed=None, render_mode=None):
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=seed)
    return env
def singleton_make_env(env_key, seed=10005, render_mode=None):
    #print(env_key)
    env = gym.make(env_key, render_mode=render_mode)
    env.reset(seed=10005)
    env = RGBImgObsWrapper(env)
    obs, _ = env.reset(seed = 10005)
    # # Save the image using PIL
    image = Image.fromarray(obs['image'])
    image.save('output_image1'+'.png')
    return env