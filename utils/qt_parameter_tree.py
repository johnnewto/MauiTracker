# -*- coding: utf-8 -*-
"""
This example demonstrates the use of pyqtgraph's parametertree system.

"""
import sys

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree, ParameterItem, registerParameterType
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QMainWindow, QAction, qApp, QApplication, QWidget
from PyQt5 import QtCore, QtGui, QtWidgets
import utils.parameters as pms
import pyqtgraph.parametertree.parameterTypes as pTypes

import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class TopWindow(QWidget):

    closed = pyqtSignal()
    def __init__(self):
        super().__init__()

    def closeEvent(self, event):
        logger.info('Close Event')
        self.closed.emit()


## test subclassing parameters
## This parameter automatically generates two child parameters which are always reciprocals of each other
class ComplexParameter(pTypes.GroupParameter):
    def __init__(self, **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)

        self.addChild({'name': 'A = 1/B', 'type': 'float', 'value': 7, 'suffix': 'Hz', 'siPrefix': True})
        self.addChild({'name': 'B = 1/A', 'type': 'float', 'value': 1 / 7., 'suffix': 's', 'siPrefix': True})
        self.a = self.param('A = 1/B')
        self.b = self.param('B = 1/A')
        self.a.sigValueChanged.connect(self.aChanged)
        self.b.sigValueChanged.connect(self.bChanged)

    def aChanged(self):
        self.b.setValue(1.0 / self.a.value(), blockSignal=self.bChanged)

    def bChanged(self):
        self.a.setValue(1.0 / self.b.value(), blockSignal=self.aChanged)

class RadioParameter(pTypes.GroupParameter):
    def __init__(self, radiobuttons, settrue=None,  **opts):
        opts['type'] = 'bool'
        opts['value'] = True
        pTypes.GroupParameter.__init__(self, **opts)
        self.handles = []
        for rb in radiobuttons:
            self.addChild({'name': rb, 'type': 'bool', 'value': False})
            self.handles.append(self.param(rb))
            self.handles[-1].sigValueChanged.connect(self.Changed)
        if settrue is None:
            self.handles[0].setValue(True)
        else:
            self.handles[settrue].setValue(True)

    def Changed(self, param, changes):
        for h in self.handles:
            h.setValue(False, blockSignal=self.Changed)
        param.setValue(True, blockSignal=self.Changed)



class ParamTree(QObject):
    params = [

        RadioParameter(['Plane','Bird','Hanglider','Paraglider'], settrue=2, name='Radio button group'),

        {'name': 'Display', 'type': 'group', 'children': [
            {'name': 'Camera Num', 'type': 'int', 'value': 0, 'limits': (0, 3), 'default': 0},
            {'name': 'Reset Fruit ID', 'type': 'action', 'tip': 'Set Fruit ID back to 1'},
            {'name': 'Scroll Image Num', 'type': 'int', 'value': 0, 'limits': (0, 10), 'default': 0},
            {'name': 'Save Images', 'type': 'action', 'tip': 'Save Scroll Image'},
            {'name': 'Plot Centers', 'type': 'action', 'tip': 'Plot the fruit centers'},
            {'name': 'Blob Tracker', 'type': 'int', 'value': 0, 'limits': (0, 1), 'default': 0},
            # {'name': 'Display Timer', 'type': 'int', 'value': pms.DISPLAY_TIMER, 'limits': (10, 100), 'default': pms.DISPLAY_TIMER},
        ]},
        {'name': 'Basic parameter data types', 'type': 'group', 'children': [
            {'name': 'Integer', 'type': 'int', 'value': 5},
            {'name': 'Float', 'type': 'float', 'value': 10.5, 'step': 0.1, 'finite': False},
            {'name': 'String', 'type': 'str', 'value': "hi", 'tip': 'Well hello'},
            {'name': 'List', 'type': 'list', 'values': [1, 2, 3], 'value': 2},
            {'name': 'Named List', 'type': 'list', 'values': {"one": 1, "two": "twosies", "three": [3, 3, 3]}, 'value': 2},
            {'name': 'Boolean', 'type': 'bool', 'value': True, 'tip': "This is a checkbox"},
            {'name': 'Color', 'type': 'color', 'value': "#FF0", 'tip': "This is a color button"},
            {'name': 'Gradient', 'type': 'colormap'},
            {'name': 'Subgroup', 'type': 'group', 'children': [
                {'name': 'Sub-param 1', 'type': 'int', 'value': 10},
                {'name': 'Sub-param 2', 'type': 'float', 'value': 1.2e6},
            ]},
            {'name': 'Text Parameter', 'type': 'text', 'value': 'Some text...'},
            {'name': 'Action Parameter', 'type': 'action', 'tip': 'Click me'},
        ]},
    ]

    def __init__(self):
        QObject.__init__(self)
        self.app = pg.mkQApp("QT Camera")
        self.win = TopWindow()
        # self.win = QtGui.QWidget()
        # self.centralwidget = QtGui.QWidget(self.win)
        self.win.setWindowTitle("Parameter Tree")

        ## Create tree of Parameter objects
        self.p = Parameter.create(name='params', type='group', children=ParamTree.params)
        self.p.sigTreeStateChanged.connect(self.change)
        # Too lazy for recursion:
        for child in self.p.children():
            child.sigValueChanging.connect(self.valueChanging)
            for ch2 in child.children():
                ch2.sigValueChanging.connect(self.valueChanging)
        # self.p.param('Save/Restore functionality', 'Save State').sigActivated.connect(self.save)
        # self.p.param('Save/Restore functionality', 'Restore State').sigActivated.connect(self.restore)

        ## Create two ParameterTree widgets, both accessing the same data
        self.pt = ParameterTree()
        self.pt.setParameters(self.p, showTop=False)
        self.pt.setWindowTitle('pyqtgraph example: Parameter Tree')

        self.layout = QtWidgets.QGridLayout()
        self.win.setLayout(self.layout)
        self.layout.addWidget(QtWidgets.QLabel("View of the  data."), 0, 0, 1, 1)
        self.layout.addWidget(self.pt, 1, 0, 1, 1)

    ## If anything changes in the tree, print a message
    def change(self, param, changes):
        print("tree changes:")


    def valueChanging(self, param, value):
        print("Value changing (not finalized): %s %s" % (param, value))

    def save(self):
        global state
        state = self.p.saveState()

    def restore(self):
        global state
        add = self.p['Save/Restore functionality', 'Restore State', 'Add missing items']
        rem = self.p['Save/Restore functionality', 'Restore State', 'Remove extra items']
        self.p.restoreState(state, addChildren=add, removeChildren=rem)


    def show(self, main_app=False):
        self.win.show()
        if main_app:
            try:
                from IPython.lib.guisupport import start_event_loop_qt4
                start_event_loop_qt4(self.app)
            except ImportError:
                self.app.exec_()

    def param_change(self, param, changes):
        logger.info(f"Paramater change ")
        p = self.p
        for c in p.children():
            if c.opts['name'] == 'Radio button group':
                if hasattr(c, 'children'):
                    for k in c.children():
                        if k.opts['name'] == param:
                            k.setValue(True)
                        else:
                            k.setValue(False)

        # logger.info(f"Paramater change {param.name()}: {param.value()}: ")
        # if param.name() == 'Plane':
        #     print('Plane')
        #     pass

    def param_change_full(self, param, changes):
        for param, change, data in changes:
            logger.info(f"param_change_full {param.name()}: {param.value()}: ")
            if param.name() == 'Plane':
                print('Plane')

                pass

            if param.name() == 'Enable':
                if param.value():
                    self.timer_start()
                else:
                    self.timer_stop()
            if param.name() == 'Camera Num':
                num = min(len(self.ftracks)-1, param.value())
                self.ftrack = self.ftracks[num]

            if param.name() == 'Blob Tracker':
                num = min(len(self.ftracks[0].trkBlob)-1, param.value())
                self.trkBlobnum = num

            if param.name() == 'Calibrate Cams':
                self.calibrateCameras()
            # if param.name() == 'Fruit Speed':
            #     self.ftrack.cam.set_speed(param.value())
            if param.name() == 'Skip Images':
                self.ftrack.cam.cam.skip = param.value()
            if param.name() == 'Grab Width':
                self.cup_trk.scrollImages.grab_width = param.value()
            if param.name() == 'Scroll Reverse Endian':
                self.ftrack.scrollImages.reverse_endian = param.value()

            elif param.name() == 'Number Fruit':
                self.ftrack.cam.set_num_fruit(param.value())
            elif param.name() == 'Reset Fruit ID':
                # self.timer_stop()
                self.cup_trk.reset()
                self.trackers.fgen.reset()
                # for ft in self.ftracks:
                #     ft.reset()
                #     img = cv2.hconcat(ft.fullImages.image_stack[self.scroll_num])
                #     print('Reset Fruit ID', img.max())
                # # self.timer_start()
                # # todo on reset with left or right it pauses for a second or so
                # print()
            elif param.name() == 'Scroll Image Num':
                self.scroll_num = param.value()
                self.plotscroll()
            elif param.name() == 'Save Images':
                self.saveImages()
            elif param.name() == 'Plot Centers':
                self.plotCenters()

if __name__ == '__main__':
    params = ParamTree()
    params.show(main_app=True)

