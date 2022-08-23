
import pyqtgraph as pg
import sys

from PyQt5.QtCore import QEvent, Qt
from PyQt5.QtWidgets import QPushButton
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
# from PyQt5 import QtCore, QtGui
import numpy as np
from utils.image_utils import *
from utils.g_images import *
from utils import parameters as pms
from utils.horizon import find_sky_2
import cv2
import csv
from pathlib import Path

import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)


def pointsAt(scat, pos):
    return scat.points()[_maskAt(scat, pos)][::-1]


def _maskAt(scat, obj):
    """
    Return a boolean mask indicating all points that overlap obj, a QPointF or QRectF.
    """
    if isinstance(obj, QtCore.QPointF):
        l = r = obj.x()
        t = b = obj.y()
    elif isinstance(obj, QtCore.QRectF):
        l = obj.left()
        r = obj.right()
        t = obj.top()
        b = obj.bottom()
    else:
        raise TypeError

    if scat.opts['pxMode'] and scat.opts['useCache']:
        w = scat.data['sourceRect']['w']
        h = scat.data['sourceRect']['h']
    else:
        s, = scat._style(['size'])
        w = h = s

    w = w * 10
    h = h * 10

    if scat.opts['pxMode']:
        # determine length of pixel in local x, y directions
        px, py = scat.pixelVectors()
        try:
            px = 0 if px is None else px.length()
        except OverflowError:
            px = 0
        try:
            py = 0 if py is None else py.length()
        except OverflowError:
            py = 0
        w *= px
        h *= py

    return (scat.data['visible']
            & (scat.data['x'] + w > l)
            & (scat.data['x'] - w < r)
            & (scat.data['y'] + h > t)
            & (scat.data['y'] - h < b))



import weakref
class Graph(pg.GraphItem):
    sigCurveHovered = QtCore.Signal(object, object)
    sigCurveNotHovered = QtCore.Signal(object, object)
    def __init__(self, parent, name=None):
        # self.parent = weakref.ref(parent)  # <= garbage-collector safe!
        self.parent = parent  # <= garbage-collector safe!
        self.name = name
        self.dragPoint = None
        self.dragOffset = None
        self.downPos = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)
        self.dataChanged = False
        self.menu = None
        self._mouseShape = None
        # self.scatter.opts['pxMode'] = False
        # self.hoverable = True
        # self.setAcceptHoverEvents(True)
        # self.sigCurveHovered.connect(self.hovered)
        # self.sigCurveNotHovered.connect(self.leaveHovered)
        self.hoverable = True
        self.hoverPen = pg.mkPen('g')
        self.hoverSize = 1e-6


    # def hovered(self):
    #     self.data['symbolPen'] = pg.mkPen('b', width=10)
    #     self.updateGraph()
    #
    # def leaveHovered(self):
    #     # self.setPen(pg.mkPen('w', width=10))
    #     self.data['symbolPen'] = pg.mkPen('w', width=10)
    #     self.updateGraph()
    #
    #
    #
    # def hoverEvent(self, ev):
    #     if self.hoverable:
    #         pts = pointsAt(self.scatter, ev.pos())
    #         if len(pts) > 0:
    #             self.sigCurveHovered.emit(self, ev)
    #         else:
    #             self.sigCurveNotHovered.emit(self, ev)

    def setData(self, **kwds):
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', int)])
            self.data['data']['index'] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def setDataChanged(self, _bool):
        self.setGraphColour(_bool)
        # do this after changing the colour
        self.dataChanged = _bool

    def getDataChanged(self, name=None):
        return self.dataChanged

    def setGraphColour(self, _bool, name=None):
        if _bool:
            # self.graph.pen = pg.mkPen('r', width=2)
            # self.graph.symbolPen = pg.mkPen('r', width=2)
            self.data['pen'] = pg.mkPen('r', width=2)
            self.data['symbolPen'] = pg.mkPen('r', width=10)
        else:
            # self.graph.pen = pg.mkPen('g', width=2)
            # self.graph.symbolPen = pg.mkPen('g', width=2)
            self.data['pen'] = pg.mkPen('g', width=2)
            self.data['symbolPen'] = pg.mkPen('g', width=10)

        self.updateGraph()

    def updateGraph(self):
        # it's essential to call this to convert the self.data into the graph points
        pg.GraphItem.setData(self, **self.data)

        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        self.dataChanged = True

    # On right-click, raise the context menu
    def mouseClickEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.raiseContextMenu(ev):
                ev.accept()

    def raiseContextMenu(self, ev):
        self.downPos = ev.pos()
        menu = self.getMenu()
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(pos.x(), pos.y()))
        return True


    def graph_addpoint(self):
        print("graph_addpoint", self.downPos)
        data = self.data
        pos = list(data['pos'])
        data['pos'] = np.append(data['pos'], [[128,128]], axis=0)
        data['adj'][-1] = np.array([3,4])
        data['adj'] = np.append(data['adj'], [[4,0]], axis=0)
        pass
        # n_points = pos.shape[0]
        # (cols, rows) = self.display_image.shape[:2]
        # pos[:,0] = pos[:,0] * cols
        # pos[:,1] = pos[:,1] * rows
        # assert max(pos[:,0]) < cols , 'columns must be less than image width '
        # adj = np.stack([np.arange(n_points), np.arange(1, n_points+1)]).transpose()
        # adj[-1,1] = 0
        # # pen = pg.mkPen('g', width=2)
        # # self.graph.setData(pos=pos, adj=adj, size=10, pen=pen, symbol='o', symbolPen=pen, symbolBrush=(50, 50, 200, 00), pxMode=False)
        # done = False
        #
        # graph.setData(pos=pos, adj=adj, size=10, symbol='o', symbolBrush=(50, 50, 200, 00), pxMode=False)



    def graph_delpoint(self):
        print("graph_delpoint", self.downPos)


    def graph_delregion(self):
        print("graph_delregion", self.downPos)

    # # On right-click, raise the context menu
    # def mouseClickEvent(self, ev):
    #     if ev.button() == QtCore.Qt.RightButton:
    #         if self.raiseContextMenu(ev):
    #             ev.accept()

    def getMenu(self):
        """
        Create the menu
        """
        if self.menu is None:
            self.menu = QtGui.QMenu()
            self._addpoint = QtGui.QAction("Add point", self.menu)
            self._addpoint.triggered.connect(self.graph_addpoint)
            self.menu.addAction(self._addpoint)
            self._delpoint = QtGui.QAction("Delete Point", self.menu)
            self.menu.addAction(self._delpoint)
            self._delpoint.triggered.connect(self.graph_delpoint)
            self._delregion = QtGui.QAction("Delete Region", self.menu)
            self._delregion.triggered.connect(self.graph_delregion)
            self.menu.addAction(self._delregion)
        return self.menu



    def mouseDragEvent(self, ev):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            shiftkey = True
        else:
            shiftkey = False
        if modifiers == QtCore.Qt.ControlModifier:
            ctrlkey = True
        else:
            ctrlkey = False

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():

            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            self.downPos = ev.buttonDownPos()
            # pts = self.scatter.pointsAt(self.downPos)
            pts = pointsAt(self.scatter, self.downPos)

            if len(pts) == 0:
                ev.ignore()
                return
            self.dragPoint = pts[0]
            ind = pts[0].data()[0]
            self.dragOffset = self.data['pos'][ind] - self.downPos
            self.dragOffsets = []
            self.startPoints = []
            for pnt in self.data['pos']:
                self.dragOffsets.append(pnt - self.downPos)
                self.startPoints.append(pnt)

        elif ev.isFinish():
            self.dragPoint = None
            return
        else:
            if self.dragPoint is None:
                ev.ignore()
                return

        if shiftkey:
            # move all the points
            ind = self.dragPoint.data()[0]
            n_points = len(self.data['pos'])
            dragAmount = [None] * 2
            if ind == 0 or ind == n_points-1:
                # rotate about the end point
                dragAmount[0] = ev.pos()[0] + self.dragOffsets[ind][0] - self.startPoints[ind][0]
                dragAmount[1] = ev.pos()[1] + self.dragOffsets[ind][1] - self.startPoints[ind][1]
                for i, (pnt, s_pnt) in enumerate(zip(self.data['pos'], self.startPoints)):
                    if ind == 0:
                        if pms.DRAG_MODE == 'XY':
                            pnt[0] = self.startPoints[i][0] + dragAmount[0] * (n_points-i)/n_points
                        pnt[1] = self.startPoints[i][1] + dragAmount[1] * (n_points-i)/n_points
                    else:
                        if pms.DRAG_MODE == 'XY':
                            pnt[0] = self.startPoints[i][0] + dragAmount * i/n_points
                        pnt[1] = self.startPoints[i][1] + dragAmount * i/n_points
            else:
                # move neighbour points by 1/2
                dragAmount[0] = ev.pos()[0] + self.dragOffsets[ind][0] - self.startPoints[ind][0]
                dragAmount[1] = ev.pos()[1] + self.dragOffsets[ind][1] - self.startPoints[ind][1]
                if pms.DRAG_MODE == 'XY':
                    self.data['pos'][ind][0] += dragAmount[0]
                    self.data['pos'][ind-1][0] += dragAmount[0]*0.5
                    self.data['pos'][ind+1][0] += dragAmount[0]*0.5
                self.data['pos'][ind][1] += dragAmount[1]
                self.data['pos'][ind-1][1] += dragAmount[1]*0.5
                self.data['pos'][ind+1][1] += dragAmount[1]*0.5

        if ctrlkey:
                # move all the same
                for i, pnt in enumerate(self.data['pos']):
                    if pms.DRAG_MODE == 'XY':
                        pnt[0] = ev.pos()[0] + self.dragOffsets[i][0]
                    pnt[1] = ev.pos()[1] + self.dragOffsets[i][1]
        else:
            # move one point only
            ind = self.dragPoint.data()[0]
            if pms.DRAG_MODE == 'XY':
                self.data['pos'][ind][0] = ev.pos()[0] + self.dragOffsets[ind][0]
            self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffsets[ind][1]

        self.data['pen'] = pg.mkPen('r', width=2)
        self.data['symbolPen'] = pg.mkPen('r', width=10)
        self.updateGraph()
        ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)


