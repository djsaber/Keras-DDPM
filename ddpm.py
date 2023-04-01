# coding=gbk

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class DDPM():
    '''DDPM模型实现
    参数:
        - network：去噪网络
        - step：去噪步数，1000
        - alpha：噪声权重数组
    方法:
        - sample()：采样
    '''

    def __init__(self, network, steps=None, alpha=None):
        self.net = network
        self.steps = steps if steps else 1000
        self.alpha = alpha if alpha else 1 - 0.02*np.arange(1, self.steps+1)/self.steps
        self.bar_alpha = np.cumprod(self.alpha)


    def sample(self, batch, size, save_path=None):
        z_samples = np.random.randn(batch, *size)
        for t in tqdm(reversed(range(self.steps)), desc="generating image"):
            bt = np.array([t for _ in range(batch)])
            noise_predict = self.net([z_samples, bt])
            z_samples = (1/np.sqrt(self.alpha[t])) * (z_samples-(((1-self.alpha[t])/(np.sqrt(1-self.bar_alpha[t])))*noise_predict))
            z_samples += np.sqrt(1-self.alpha[t]) * np.random.randn(batch, *size)

        imgs_arr = np.clip(z_samples, -1, 1)
        if save_path:
            imgs_arr = imgs_arr * 0.5 + 0.5
            for i in range(batch):
                plt.imsave(f'{save_path}{i}.png', imgs_arr[i])