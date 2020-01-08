import sys
import cv2
import skimage.io
import numpy as np
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import QDialog, QFileDialog, QFrame, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QImage

import vtk
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw

import qimage2ndarray
fname = ""
    
class Renderer(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("renderer.ui",self)

        self.setWindowTitle("Reinforcement Volume Rendering")
        self.pbOpen.clicked.connect(self.display)
        # self.pbGrayImage.clicked.connect(self.convert_to_gray)

    def display(self):
        global fname
        fname = QFileDialog.getOpenFileName(self, 'Open file...', '/home/tmquan/iRenQt/',"Image Files (*.jpg *.tif *.bmp *.png)")
        print(fname)
        data = skimage.io.imread(fname[0])
        print(data.shape)
        dimz, dimy, dimx = data.shape
        dataXY = data[int(dimz/2),:,:]
        dataXZ = data[:,int(dimy/2),:]
        dataYZ = data[:,:,int(dimx/2)]

        self.labelXY.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(dataXY)))
        self.labelXZ.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(dataXZ)))
        self.labelYZ.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(dataYZ)))

        ###
        self.frame = QFrame()
 
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.layoutScreen.addWidget(self.vtkWidget)
 
        self.render = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.render) # Attach vtk renderer 
        self.renWindow = self.vtkWidget.GetRenderWindow()
        self.renWinInt = self.renWindow.GetInteractor()

        self.reader = vtkTIFFReader()
        self.reader.SetFileName(fname[0])
        self.reader.Update()
        
        self.mapper = vtkFixedPointVolumeRayCastMapper()
        self.mapper.SetInputConnection(self.reader.GetOutputPort())
        self.mapper.SetBlendModeToMaximumIntensity()

        self.prop = vtkVolumeProperty()
        colorFunc = vtkColorTransferFunction()
        alphaFunc = vtkPiecewiseFunction()

        # Set up the Property
        colorFunc.AddRGBSegment(0.0, 0.0, 0.0, 0.0,
                                255.0, 1.0, 1.0, 1.0)
        alphaRange = 4096
        alphaLevel = 1024
        alphaFunc.AddSegment(alphaLevel - 0.5*alphaRange, 0.0,
                             alphaLevel + 0.5*alphaRange, 1.0 )
        self.prop.SetIndependentComponents(True)
        self.prop.SetColor(colorFunc)
        self.prop.SetScalarOpacity(alphaFunc)
        self.prop.SetInterpolationTypeToLinear()

        # Set up the data
        self.data = vtkVolume()
        self.data.SetMapper(self.mapper)
        self.data.SetProperty(self.prop)

        # Set up the renderer
        self.render.SetBackground(0.0, 0.0, 0.0)
        self.render.AddVolume(self.data)

        self.show()
        self.renWinInt.Initialize()
        self.renWindow.Render()
        self.renWinInt.Start()
        
if __name__=="__main__":
    app = QApplication(sys.argv)    
    win = Renderer()
    win.show()
    sys.exit(app.exec_())