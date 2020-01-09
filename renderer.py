import sys
import cv2
import skimage.io
import numpy as np
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from PyQt5.QtWidgets import*
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QDialog, QFileDialog, QFrame, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt, QPoint

import vtk
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import qimage2ndarray
  
class Renderer(QMainWindow):

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi("renderer.ui",self)

        self.setWindowTitle("Reinforcement Volume Rendering")
        self.pbOpen.clicked.connect(self.load_data)
        self.pbOpen.clicked.connect(self.display_editor)
        self.pbOpen.clicked.connect(self.display_slices)
        self.pbOpen.clicked.connect(self.display_screen)

    def load_data(self):
        self.fname = QFileDialog.getOpenFileName(self, 'Open file...', '/home/tmquan/iRenQt/',"Image Files (*.jpg *.tif *.bmp *.png)")
        print(self.fname)
        self.data = skimage.io.imread(self.fname[0])
        print(self.data.shape)

    def display_editor(self):        
        self.canvas = QPixmap(409, 209)
        self.canvas.fill(Qt.black)
        self.labelEditor.setPixmap(self.canvas)

    def display_slices(self):
        dimz, dimy, dimx = self.data.shape
        dataXY = self.data[int(dimz/2),:,:]
        dataXZ = self.data[:,int(dimy/2),:]
        dataYZ = self.data[:,:,int(dimx/2)]

        self.labelXY.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(dataXY)))
        self.labelXZ.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(dataXZ)))
        self.labelYZ.setPixmap(QPixmap.fromImage(qimage2ndarray.array2qimage(dataYZ)))

    def display_screen(self):
        self.frame = QFrame()
 
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.layoutScreen.addWidget(self.vtkWidget)
 
        self.render = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.render) # Attach vtk renderer 
        self.renWindow = self.vtkWidget.GetRenderWindow()
        self.renWinInt = self.renWindow.GetInteractor()

        self.reader = vtkTIFFReader()
        self.reader.SetFileName(self.fname[0])
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

        # self.show()
        self.renWinInt.Initialize()
        # self.renWindow.Render()
        self.renWinInt.Start()
        
if __name__=="__main__":
    app = QApplication(sys.argv)    
    win = Renderer()
    win.show()
    sys.exit(app.exec_())