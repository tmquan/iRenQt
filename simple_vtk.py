#!/usr/bin/env python
 
import sys
import argparse
import vtk
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw

# import numpy as np
# import cv2

class MainWindow(qtw.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.image = args.image
        self.frame = qtw.QFrame()
 
        self.layout = qtw.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtkWidget)
 
        self.render = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.render) # Attach vtk renderer 
        self.renWindow = self.vtkWidget.GetRenderWindow()
        self.intWindow = self.renWindow.GetInteractor()

        self.reader = vtkTIFFReader()
        self.reader.SetFileName(self.image)
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
        self.render.SetBackground(0.5,0.5,0.5)
        self.render.AddVolume(self.data)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
    
        self.show()
        self.intWindow.Initialize()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Volume *.tif file')
    args = parser.parse_args()
    print(args)
    
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())