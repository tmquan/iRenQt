#!/usr/bin/env python
 
import sys
import argparse
import vtk
from vtk import *
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PyQt5 import QtCore as qtc
from PyQt5 import QtWidgets as qtw


class MainWindow(qtw.QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

        self.image = args.image
        self.frame = qtw.QFrame()
 
        self.layout = qtw.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.layout.addWidget(self.vtkWidget)
 
        self.render = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.render)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.reader = vtkTIFFReader()
        self.mapper = vtkFixedPointVolumeRayCastMapper()

        self.reader.SetFileName(self.image)
        self.reader.Update()
        
        self.mapper.SetInputConnection(self.reader.GetOutputPort())
        self.mapper.SetBlendModeToMaximumIntensity()

        prop = vtkVolumeProperty()
        data = vtkVolume()

        colorFunc = vtkColorTransferFunction()
        alphaFunc = vtkPiecewiseFunction()

        # Set up the Property
        colorFunc.AddRGBSegment(0.0, 0.0, 0.0, 0.0,
                                255.0, 1.0, 1.0, 1.0)
        alphaRange = 4096
        alphaLevel = 1024
        alphaFunc.AddSegment(alphaLevel - 0.5*alphaRange, 0.0,
                             alphaLevel + 0.5*alphaRange, 1.0 )
        prop.SetIndependentComponents(True)
        prop.SetColor(colorFunc)
        prop.SetScalarOpacity(alphaFunc)
        prop.SetInterpolationTypeToLinear()

        # Set up the data
        data.SetMapper(self.mapper)
        data.SetProperty(prop)

        # Set up the renderer
        self.render.SetBackground(0,0,0)
        self.render.AddVolume(data)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
 
        self.show()
        self.iren.Initialize()
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='tif file')
    args = parser.parse_args()
    print(args)
    app = qtw.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())