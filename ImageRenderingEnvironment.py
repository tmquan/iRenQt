from natsort import natsorted
import os
import glob
import glob2
import cv2
import skimage.io
import skimage.measure
import sklearn
import sklearn.metrics
from scipy import signal
import numpy as np
from PIL import Image
# Using torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weightNorm
import torch.optim as optim
from torch.optim.optimizer import Optimizer, required
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
import torchvision
import gym

from environment.BaseEnvironment import BaseEnvironment

from skimage.metrics import structural_similarity as ssim

def prob2size(prob, size):
    # Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
    # prob from [0 to 1) -> size [0 to size)
    # prob2size(0.8, 512)     409
    # prob2size(1.0, 512)     511
    # prob2size(0.0, 512)     0
    return (int)(prob * (size - 1) + 0.5)

def prob2prob(prob, low, high):
    # Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
    # prob from [0 to 1) -> size [0 to size)
    return (prob * (high-low) + low) 

SHAPE = 256
class ImageRenderingEnvironment(BaseEnvironment):
    def __init__(self, 
        max_step_per_episode=30, 
        target=None, 
        volume=None, 
        writer=None):
        """Environment for an agent can interact to adjust the lookup table
        
        [description]
        
        Arguments:
            **kwargs {[type]} -- [description]
        """
        self.max_step_per_episode = max_step_per_episode
        self.writer = writer
        # assert target is not None and volume is not None
        self.target = target if target is not None else np.zeros((SHAPE, SHAPE, 4), dtype=np.uint8) # RGB Image
        self.volume = volume if volume is not None else np.zeros((SHAPE, SHAPE, 1), dtype=np.uint8) # Gray Image to construct the lookup table
        
        self.screen = None 
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,)) # {Pixel Value: R G B A}
        self.global_step = 0
        self.global_episode = 0

        self.reset()

    

    def reset(self):
        print("="*80)
        print("Environment has been reset...")
        print("="*80)

        zeros = np.zeros(256, np.dtype('uint8'))
        self.LTable = np.stack((zeros, zeros, zeros, zeros), 1)
        self.screen = np.zeros_like(self.target) # RGB Image to be painted, 
        self.curr_rwd = 0 
        self.prev_rwd = self.curr_rwd
        self.trial = 0

        # self.target[:] = np.random.randint(0, 256, 4)
        # self.volume[:] = np.random.randint(0, 256, 1)

        obs = np.concatenate([self.volume / 255.0, self.target / 255.0, self.screen / 255.0 ], -1)
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs[np.newaxis,...]
        print(self.volume[0,0], self.target[0,0],  self.screen.shape, self.target.shape, self.LTable.shape, obs.shape)

        return obs

    def step(self, action):
        # Need to return obs, rwd, done, info 
        obs = None
        rwd = None
        done = None
        info = None

        
        Dvalue = prob2size(action[0], 256) # Data value
        Rvalue = prob2size(action[1], 256)
        Gvalue = prob2size(action[2], 256)
        Bvalue = prob2size(action[3], 256)
        Avalue = prob2size(action[4], 256)

        self.LTable[Dvalue] = np.array([Rvalue, Gvalue, Bvalue, Avalue]) # TODO

        # Ray cast the data with new lookup table       
        # self.screen = np.zeros_like(self.target)  
        # print(self.LTable)
        self.screen = self.LTable[tuple(self.volume.transpose())] #.transpose()

        # Update the reward
        score = ssim(self.screen, self.target, multichannel=True)
        self.curr_rwd = score
        rwd = self.curr_rwd - self.prev_rwd

        # Update flags
        done = True if self.trial == self.max_step_per_episode else False
        self.global_step += 1
        self.trial += 1
        self.prev_rwd = self.curr_rwd.copy()

        obs = np.concatenate([self.volume / 255.0, self.target / 255.0, self.screen / 255.0 ], -1)
        obs = np.transpose(obs, (2, 0, 1))
        obs = obs[np.newaxis,...]
        # print(self.screen.shape, self.target.shape, obs.shape)
        # cv2.imshow('', np.concatenate([cv2.cvtColor(self.volume, cv2.COLOR_GRAY2RGBA), self.screen, self.target], 1))
        # cv2.waitKey(10)
        print('Local step {}\t, Action is {}\t, Reward is {:0.5f}\t, Score is {:0.5f}\t    {}'.format(self.trial, [Dvalue, Rvalue, Gvalue, Bvalue, Avalue], rwd, score, done))
        return obs, rwd, done, info


if __name__ == '__main__':
    writer = SummaryWriter()
    target = np.zeros((256 ,256, 4), np.uint8)
    volume = np.zeros((256 ,256, 1), np.uint8)

    target[:] = [255, 255, 0, 0]
    volume[:] = [120]
    env = ImageRenderingEnvironment(writer=writer, target=target, volume=volume)

    # if args.mode == 'random':
    np.random.seed(2222)
    obs, rwd, done, info = env.step([0, 0, 0, 0, 0])
    for _ in range(100):
        act = np.random.uniform(0, 1, 5)
        obs, rwd, done, info = env.step(act)
