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
from models.uppnet import UPPNet

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


class ImageRenderingEnvironment(BaseEnvironment):
    def __init__(self, **kwargs):
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(5,))
        self.global_step = 0
        self.global_episode = 0
        self.max_step_per_episode = max_step_per_episode
        self.writer = writer

        self.reset()

    

    def reset(self):
        pass

    def step(self, action):
        # Need to return obs, rwd, done, info 
        obs = None
        rwd = None
        done = None
        info = None


        # Update flags
        self.global_step += 1
        self.trial += 1
        self.prev_rwd = self.curr_rwd
        print('Local step {}, Action is {}, Reward is {}, Rand score is {}'.format(self.trial, action, rwd, rand_score))
        return obs, rwd, done, info


if __name__ == '__main__':
    writer = SummaryWriter()
    env = ImageRenderingEnvironment(writer=writer)

    # if args.mode == 'random':
    np.random.seed(2222)
    obs, rwd, done, info = env.step([0, 0, 0, 0])
    for _ in range(100):
        act = np.random.uniform(0, 1, 4)
        # print(act)
        obs, rwd, done, info = env.step(act)
        print(done)
