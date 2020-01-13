from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import glob
import math
import random
import shutil
import logging
import argparse
from itertools import count
from natsort import natsorted
import glob2
import cv2
import skimage.io
import skimage.measure
import skimage.segmentation
import sklearn
import sklearn.metrics
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

from optim.RAdam import RAdam
from environment.BaseEnvironment import BaseEnvironment
from ImageRenderingEnvironment import ImageRenderingEnvironment

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=500000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        x = np.array(x)
        y = np.array(y)
        u = np.array(u)
        r = np.array(r)
        d = np.array(d)

        x = np.squeeze(x, axis=1)
        y = np.squeeze(y, axis=1)
        # print(x.shape, y.shape, u.shape, r.shape, d.shape)
        return x, y, u, r, d

class Actor(nn.Module):
    def __init__(self, num_classes=5):
        super(Actor, self).__init__()
        self.model = getattr(torchvision.models, 'densenet121')(pretrained=False)
        self.model.features.conv0 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
            )

    def forward(self, x):
        logit = self.model(x / 127.5 - 1.0)
        return logit

class Critic(nn.Module):
    def __init__(self, num_classes=1):
        super(Critic, self).__init__()
        self.model = getattr(torchvision.models, 'densenet121')(pretrained=False)
        self.model.features.conv0 = nn.Conv2d(11, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.Sigmoid()
            )

    def forward(self, x, y):
        # print(y.shape)
        y = y.reshape(y.shape[0], y.shape[1], 1, 1)
        y = F.upsample(y, size=(256, 256), mode='nearest')
        # print(y.shape)
        logit = self.model(torch.cat([x / 127.5 - 1.0, y], 1))
        return logit

class DDPG(object):
    def __init__(self, writer=None, device='cuda', hparams=None):
        self.device = device
        self.hparams = hparams
        self.actor = Actor(num_classes=5).to(self.device)
        self.actor_target = Actor(num_classes=5).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = RAdam(self.actor.parameters(), lr=self.hparams.lr, 
            betas=(0.9, 0.99), weight_decay=1e-4)


        self.critic = Critic(num_classes=1).to(self.device)
        self.critic_target = Critic(num_classes=1).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = RAdam(self.actor.parameters(), lr=self.hparams.lr, 
            betas=(0.9, 0.99), weight_decay=1e-4)
        self.replay_buffer = Replay_buffer(max_size=self.hparams.capacity)
        self.writer = writer #SummaryWriter()
        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        # print(state.shape)
        return self.actor(state).detach().cpu().numpy().flatten()

    def update(self):

        for it in range(self.hparams.update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self.hparams.batch_size)
            state = torch.FloatTensor(x).to(self.device)
            action = torch.FloatTensor(u).to(self.device)
            next_state = torch.FloatTensor(y).to(self.device)
            done = torch.FloatTensor(d).to(self.device)
            reward = torch.FloatTensor(r).to(self.device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + ((1 - done) * self.hparams.gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.hparams.tau * param.data + (1 - self.hparams.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.hparams.tau * param.data + (1 - self.hparams.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), 'actor.pth')
        torch.save(self.critic.state_dict(), 'critic.pth')
        print("====================================")
        print("Model has been saved...")
        print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load('actor.pth'))
        self.critic.load_state_dict(torch.load('critic.pth'))
        print("====================================")
        print("Model has been loaded...")
        print("====================================")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default=0, type=int, help='comma separated list of GPU(s) to use.')
    parser.add_argument('--mode', default='train', type=str) # mode = 'train' or 'test'
    # OpenAI gym environment name, # ['BipedalWalker-v2', 'Pendulum-v0'] or any continuous environment
    # Note that DDPG is feasible about hyper-parameters.
    # You should fine-tuning if you change to another environment.
    # parser.add_argument("--env_name", default="Pendulum-v0")
    parser.add_argument('--tau',  default=0.005, type=float) # target smoothing coefficient
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)

    parser.add_argument('--learning_rate', '-lr', default=2e-4, type=float,
                            metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--gamma', default=0.95, type=float) # discounted factor
    parser.add_argument('--capacity', default=1000, type=int) # replay buffer size
    parser.add_argument('--batch_size', default=4, type=int) # mini batch size
    parser.add_argument('--seed', default=True, type=bool)
    parser.add_argument('--random_seed', default=2222, type=int)
    # optional parameters

    parser.add_argument('--sample_frequency', default=256, type=int)
    parser.add_argument('--render', default=False, type=bool) # show UI or not
    parser.add_argument('--log_interval', default=50, type=int) #
    # parser.add_argument('--load', default=False, type=bool) # load model
    parser.add_argument('--load', action='store_true') # load model
    parser.add_argument('--render_interval', default=100, type=int) # after render_interval, the env.render() will work
    parser.add_argument('--exploration_noise', default=0.4, type=float)
    parser.add_argument('--max_episode', default=100000, type=int) # num of games
    parser.add_argument('--max_step_per_episode', default=50, type=int) # num of games
    parser.add_argument('--max_length_of_trajectory', default=2000, type=int) # num of games
    parser.add_argument('--print_log', default=1, type=int)
    parser.add_argument('--update_iteration', default=10, type=int)
    return parser.parse_args()


def main(hparams):
    if hparams.seed is not None:
        random.seed(hparams.seed)
        np.random.seed(hparams.seed)
        torch.manual_seed(hparams.seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(hparams.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    use_cuda = torch.cuda.is_available()
    xpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if hparams.load is not None: # Load the checkpoint here\
        pass


    writer = SummaryWriter()
    target = np.zeros((256 ,256, 3), np.uint8)
    volume = np.zeros((256 ,256, 1), np.uint8)

    target[:] = [255, 255, 0]
    volume[:] = [120]
    env = ImageRenderingEnvironment(writer=writer, target=target, volume=volume)

    agent = DDPG(writer=writer, device=xpu, hparams=hparams)
    ep_r = 0
    if hparams.mode == 'test':
        agent.load()
        for i in range(hparams.test_iteration):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                ep_r += reward
                env.render()
                if done or t >= hparams.max_length_of_trajectory:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break
                state = next_state

    elif hparams.mode == 'train':
        print("====================================")
        print("Collection Experience...")
        print("====================================")
        if hparams.load: agent.load()
        for i in range(hparams.max_episode):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)

                # issue 3 add noise to action
                action = (action + np.random.normal(0, hparams.exploration_noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)

                next_state, reward, done, info = env.step(action)
                ep_r += reward
                if hparams.render and i >= hparams.render_interval : env.render()
                agent.replay_buffer.push((state, next_state, action, reward, np.float(done)))
                if (i+1) % 10 == 0:
                    print('Episode {},  The memory size is {} '.format(i, len(agent.replay_buffer.storage)))

                state = next_state
                if done or t >= hparams.max_length_of_trajectory:
                    agent.writer.add_scalar('ep_r', ep_r, global_step=i)
                    if i % hparams.print_log == 0:
                        print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}".format(i, ep_r, t))
                    ep_r = 0
                    break

            if i % hparams.log_interval == 0:
                agent.save()
            if len(agent.replay_buffer.storage) >= hparams.capacity-1:
                agent.update()

    else:
        raise NameError("mode wrong!!!")

if __name__ == '__main__':
    main(get_args())