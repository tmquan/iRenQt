import gym
import cv2
import skimage.io
import vtk
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw
import torch
import random
import numpy as np
from collections import deque

from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error as mse

def prob2size(prob, size):
    # Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
    # prob from [0 to 1) -> size [0 to size)
    # prob2size(0.8, 512)     409
    # prob2size(1.0, 512)     511
    # prob2size(0.0, 512)     0
    return (int)(prob * (size - 1) + 0.5)

def prob2prob(prob, low, high):
    # Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high)
    # prob from [0 to 1) -> size [low to high)
    return (prob * (high-low) + low) 

class RenderingEnvironment(gym.Env):
    def __init__(self, 
        max_step_per_episode=5, 
        num_agents=4,
        target_file=None, 
        volume_file=None, 
        shape=512,
        is_train=False):
        """[summary]
        
        [description]
        
        Keyword Arguments:
            max_step_per_episode {number} -- [description] (default: {5})
            num_agents {number} -- [description] (default: {4})
            target {[type]} -- [description] (default: {None})
            volume {[type]} -- [description] (default: {None})
            is_train {bool} -- [description] (default: {False})
        """
        self.max_step_per_episode = max_step_per_episode
        self.num_agents = num_agents
        self.is_train = is_train
        if self.is_train:
            self.writer = SummaryWriter()
        # Parsing the data
        target = skimage.io.imread(target_file)
        volume = skimage.io.imread(volume_file)
        target = cv2.cvtColor(target, cv2.COLOR_RGB2RGBA)
        screen = target.copy()

        # VTK window
       

        # Environment interativity
        agent_action_space = gym.spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        agent_observation_space = gym.spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space = list(agent_action_space for _ in range(self.num_agents))
        self.observation_space = list(agent_observation_space for _ in range(self.num_agents))

        self.target = target
        target_shape = self.target.shape
        self.screen = screen
        screen_shape = self.screen.shape

        # Resize the target and the screen to SHAPE
        agent_target = target  # RGBA
        self.action_target = list(agent_target for _ in range(self.num_agents))

        agent_screen = screen  # RGBA
        self.action_screen = list(agent_screen for _ in range(self.num_agents))

        # agent_volume = volume  # RGBA
        # self.action_volume = list(agent_volume for _ in range(self.num_agents))
        # print("Target shape: {} \t Screen shape: {} \t Volume shape: {}".format(agent_target.shape, agent_screen.shape, agent_volume.shape))

        # Once init the environment, we have to reset it first
        self.reset()
        self.global_step = 0
        self.global_episode = 0

    def reset(self):
        print("\nEnvironment has been reset...")
        return [0, 0, 0, 0]

    def step(self, action):
        return [0, 0, 0, 0]

    def render(self, mode='human'):
        #         #
        # # VTK handle
        # #
        # self.volume      = vtkVolume()
        # self.reader = vtkTIFFReader()
        # self.reader.SetFileName(volume_file)
        # self.reader.Update()
        
        # self.mapper = vtkFixedPointVolumeRayCastMapper()
        # self.mapper.SetInputConnection(self.reader.GetOutputPort())
        # self.mapper.SetBlendModeToMaximumIntensity()

        # self.prop = vtkVolumeProperty()
        # self.colorFunc = vtkColorTransferFunction()
        # self.colorFunc.AddRGBSegment(0.0,   0.0, 0.0, 0.0,
        #                              255.0, 1.0, 1.0, 1.0)
        # self.alphaFunc = vtkPiecewiseFunction()
        # self.alphaFunc.AddSegment(0.0,   0.0, 
        #                           55.0, 1.0,)
        # # Set up the Volume
        # self.volume.SetMapper(self.mapper)
        # self.volume.SetProperty(self.prop)
        # # Set up the renderer
        # self.render  = vtkRenderer()
        # self.render.SetBackground(0,0,0)
        # # Add volume
        # self.render.AddVolume(self.volume)
        # # Set up the windows
        # self.renWin  = vtkRenderWindow()
        # self.renWin.SetSize(512, 512)
        # self.renWin.AddRenderer(self.render)

        # # Set up the interactor
        # self.iren    = vtkRenderWindowInteractor()
        # self.iren.SetRenderWindow(self.renWin)

        # # Set up the camera
        # self.camera  = self.render.GetActiveCamera()
        # self.center  = self.volume.GetCenter()
        # self.camera.SetFocalPoint(self.center[0], self.center[1], self.center[2])
        # self.camera.SetPosition(self.center[0], self.center[1]-512, self.center[2])
        # self.camera.SetViewUp(0, 0, -1)

        # # # Start renderer
        # # self.renWin.Render()
        # # self.iren.Initialize()
        # # self.iren.Start()

        # # Environment visualization
        # self.mapper = vtkFixedPointVolumeRayCastMapper()
        # self.mapper.SetInputConnection(self.reader.GetOutputPort())
        # self.mapper.SetBlendModeToMaximumIntensity()

        # self.prop = vtkVolumeProperty()
        # colorFunc = vtkColorTransferFunction()
        # alphaFunc = vtkPiecewiseFunction()

        # # Set up the data
        # self.data = vtkVolume()
        # self.data.SetMapper(self.mapper)
        # self.data.SetProperty(self.prop)

        # # Set up the renderer
        # self.render.SetBackground(100, 100, 100)
        # self.render.AddVolume(self.data)

        # self.frame.setLayout(self.layout)
        # self.setCentralWidget(self.frame)
    
        # self.show()
        # self.renWinInt.Initialize()
        # self.renWindow.Render()
        # self.renWinInt.Start()
        pass

    def close(self):
        pass

if __name__ == '__main__':
    target_file = "target/Lenna.png"
    volume_file = "volume/Engine.tif"
    # target = skimage.io.imread(target_file)
    # volume = skimage.io.imread(volume_file)
    # target = cv2.cvtColor(target, cv2.COLOR_RGB2RGBA)
    env = RenderingEnvironment(
        target_file=target_file,
        volume_file=volume_file,
        is_train=True
        )

    # if args.mode == 'random':
    np.random.seed(2222)
    obs, rwd, done, info = env.step([0, 0, 0, 0])
    for _ in range(100):
        act = np.random.uniform(0, 1, 4)
        # print(act)
        obs, rwd, done, info = env.step(act)
        env.render()
        # print(done)