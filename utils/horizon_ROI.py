
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

#
class _PolyLineSegment(pg.LineSegmentROI):
    # Used internally by PolyLineROI
    def __init__(self, *args, **kwds):
        self._parentHovering = False
        pg.LineSegmentROI.__init__(self, *args, **kwds)

    def setParentHover(self, hover):
        # set independently of own hover state
        if self._parentHovering != hover:
            self._parentHovering = hover
            self._updateHoverColor()

    def _makePen(self):
        if self.mouseHovering or self._parentHovering:
            return self.hoverPen
        else:
            return self.pen

    def hoverEvent(self, ev):
        # accept drags even though we discard them to prevent competition with parent ROI
        # (unless parent ROI is not movable)
        if self.parentItem().translatable:
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        return pg.LineSegmentROI.hoverEvent(self, ev)


class HorizonROI(pg.PolyLineROI):
    r"""
    Container class for multiple connected LineSegmentROIs.
    This class allows the user to draw paths of multiple line segments.

    ============== =============================================================
    **Arguments**
    positions      (list of length-2 sequences) The list of points in the path.
                   Note that, unlike the handle positions specified in other
                   ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    closed         (bool) if True, an extra LineSegmentROI is added connecting
                   the beginning and end points.
    \**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================

    """

    def __init__(self, points, closed=False, pos=None, **args):

        if pos is None:
            pos = [0, 0]

        self.closed = closed
        self.segments = []
        pg.ROI.__init__(self, pos, size=[1, 1], **args)

        self.setPoints(points)

    def setPoints(self, points, closed=None):
        """
        Set the complete sequence of points displayed by this ROI.

        ============= =========================================================
        **Arguments**
        points        List of (x,y) tuples specifying handle locations to set.
        closed        If bool, then this will set whether the ROI is closed
                      (the last point is connected to the first point). If
                      None, then the closed mode is left unchanged.
        ============= =========================================================

        """
        if closed is not None:
            self.closed = closed

        self.clearPoints()
        # self.addRotateFreeHandle(points[0], points[-1])
        # for p in points[1:-1]:
        for p in points:
            self.addFreeHandle(p)
        # self.addRotateFreeHandle(points[-1], points[0])

        start = -1 if self.closed else 0
        for i in range(start, len(self.handles) - 1):
            self.addSegment(self.handles[i]['item'], self.handles[i + 1]['item'])

    def clearPoints(self):
        """
        Remove all handles and segments.
        """
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])

    def getState(self):
        state = pg.ROI.getState(self)
        state['closed'] = self.closed
        state['points'] = [pg.Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        state = pg.ROI.saveState(self)
        state['closed'] = self.closed
        state['points'] = [tuple(h.pos()) for h in self.getHandles() if h.typ == 'f']   # JN select typ == 'f' to ignore 'r'
        # state['points'] = [tuple(h.pos()) for h in self.getHandles()]   # JN select typ == 'f' to ignore 'r'
        return state

    def setState(self, state):
        pg.ROI.setState(self, state)
        self.setPoints(state['points'], closed=state['closed'])

    def addSegment(self, h1, h2, index=None):
        seg = _PolyLineSegment(handles=(h1, h2), pen=self.pen, hoverPen=self.hoverPen,
                               parent=self, movable=False)
        if index is None:
            self.segments.append(seg)
        else:
            self.segments.insert(index, seg)
        seg.sigClicked.connect(self.segmentClicked)
        seg.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        seg.setZValue(self.zValue() + 1)
        for h in seg.handles:
            h['item'].setDeletable(True)
            h['item'].setAcceptedMouseButtons(h[
                                                  'item'].acceptedMouseButtons() | QtCore.Qt.MouseButton.LeftButton)  ## have these handles take left clicks too, so that handles cannot be added on top of other handles

    def setMouseHover(self, hover):
        ## Inform all the ROI's segments that the mouse is(not) hovering over it
        pg.ROI.setMouseHover(self, hover)
        for s in self.segments:
            s.setParentHover(hover)

    def addHandle(self, info, index=None):
        h = pg.ROI.addHandle(self, info, index=index)
        h.sigRemoveRequested.connect(self.removeHandle)
        self.stateChanged(finish=True)
        return h

    def segmentClicked(self, segment, ev=None, pos=None):  ## pos should be in this item's coordinate system
        if ev is not None:
            pos = segment.mapToParent(ev.pos())
        elif pos is None:
            raise Exception("Either an event or a position must be given.")
        h1 = segment.handles[0]['item']
        h2 = segment.handles[1]['item']

        i = self.segments.index(segment)
        h3 = self.addFreeHandle(pos, index=self.indexOfHandle(h2))
        self.addSegment(h3, h2, index=i + 1)
        segment.replaceHandle(h2, h3)

    def removeHandle(self, handle, updateSegments=True):
        pg.ROI.removeHandle(self, handle)
        handle.sigRemoveRequested.disconnect(self.removeHandle)

        if not updateSegments:
            return
        segments = handle.rois[:]

        if len(segments) == 1:
            self.removeSegment(segments[0])
        elif len(segments) > 1:
            handles = [h['item'] for h in segments[1].handles]
            handles.remove(handle)
            segments[0].replaceHandle(handle, handles[0])
            self.removeSegment(segments[1])
        self.stateChanged(finish=True)

    def removeSegment(self, seg):
        for handle in seg.handles[:]:
            seg.removeHandle(handle['item'])
        self.segments.remove(seg)
        seg.sigClicked.disconnect(self.segmentClicked)
        self.scene().removeItem(seg)

    def checkRemoveHandle(self, h):
        ## called when a handle is about to display its context menu
        if self.closed:
            return len(self.handles) > 3
        else:
            return len(self.handles) > 2

    def paint(self, p, *args):
        pass

    def boundingRect(self):
        return self.shape().boundingRect()

    def shape(self):
        p = QtGui.QPainterPath()
        if len(self.handles) == 0:
            return p
        p.moveTo(self.handles[0]['item'].pos())
        for i in range(len(self.handles)):
            p.lineTo(self.handles[i]['item'].pos())
        p.lineTo(self.handles[0]['item'].pos())
        return p

    def getArrayRegion(self, *args, **kwds):
        return self._getArrayRegionForArbitraryShape(*args, **kwds)

    def setPen(self, *args, **kwds):
        pg.ROI.setPen(self, *args, **kwds)
        for seg in self.segments:
            seg.setPen(*args, **kwds)



