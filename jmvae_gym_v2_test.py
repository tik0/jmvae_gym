import numpy as np
import matplotlib.pyplot as plt
from jmvae_gym_v2 import JmvaeGym_v2



if __name__ == '__main__':
    env_x = JmvaeGym_v2(name = 'jmvae_gym_v2', modality = 'x', permutate = True, has_distance = True)
    env_w = JmvaeGym_v2(name = 'jmvae_gym_v2', modality = 'w', permutate = True, has_distance = True)
    env_v = JmvaeGym_v2(name = 'jmvae_gym_v2', modality = 'v', permutate = True, has_distance = True)

    envs = [env_x, env_w, env_v]

    for x in envs:
        x.sample_env()
        state, reward, done, _ = x.step(0)
        state, reward, done, _ = x.step(1)
        state, reward, done, _ = x.step(2)
        state, reward, done, _ = x.step(3)
        state, reward, done, _ = x.step(4)

