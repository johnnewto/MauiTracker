__all__ = ['Viewer']

import pyqtgraph as pg
import sys
from pyqtgraph.Qt import QtCore, QtWidgets
# from PyQt5 import QtCore, QtGui
import numpy as np
from utils.image_utils import Images
from utils import parameters as pms
from utils.horizon import find_sky_2
import cv2
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

class Graph(pg.GraphItem):
    def __init__(self):
        self.dragPoint = None
        self.dragOffset = None
        self.downPos = None
        self.textItems = []
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.clicked)

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

    def updateGraph(self):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])

    def mouseDragEvent(self, ev):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers == QtCore.Qt.ShiftModifier:
            shift = True
        else:
            shift = False

        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            ev.ignore()
            return

        if ev.isStart():

            # We are already one step into the drag.
            # Find the point(s) at the mouse cursor when the button was first
            # pressed:
            self.downPos = ev.buttonDownPos()
            pts = self.scatter.pointsAt(self.downPos)
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

        if shift:
            # move all the points
            ind = self.dragPoint.data()[0]
            n_points = len(self.data['pos'])
            if ind == 0 or ind == n_points-1:
                # rotate about the end point
                dragAmount = ev.pos()[1] + self.dragOffsets[ind][1] - self.startPoints[ind][1]
                for i, (pnt, s_pnt) in enumerate(zip(self.data['pos'], self.startPoints)):
                    if ind == 0:
                        pnt[1] = self.startPoints[i][1] + dragAmount * (n_points-i)/n_points
                    else:
                        pnt[1] = self.startPoints[i][1] + dragAmount * i/n_points


            else:
                # move all the same
                for i, pnt in enumerate(self.data['pos']):
                    pnt[1] = ev.pos()[1] + self.dragOffsets[i][1]
        else:
            # move one point only
            ind = self.dragPoint.data()[0]
            self.data['pos'][ind][1] = ev.pos()[1] + self.dragOffsets[ind][1]

        self.updateGraph()
        ev.accept()

    def clicked(self, pts):
        print("clicked: %s" % pts)


g_images = Images()
g_images.small_rgb = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)


def setCurrentImages(images, image):
    global g_images
    g_images = images
    g_images.set(image)

def getCurrentImages():
    global g_images
    return g_images

class Widget(QtWidgets.QWidget):
    def __init__(self):
        super(Widget, self).__init__()
        self.init_ui()
        self.qt_connections()
        self._key_press = None
        self.viewCurrentImage()

    def init_ui(self):
        self.setWindowTitle('Viewer')
        self.widget = pg.GraphicsLayoutWidget()
        self.img = pg.ImageItem(border='w')
        self.g = Graph()
        view = self.widget.addViewBox()
        view.addItem(self.img)
        view.addItem(self.g)

        self.autoLabel = QtWidgets.QPushButton("Auto Horizon")
        self.free1 = QtWidgets.QPushButton("Free")
        self.free2 = QtWidgets.QPushButton("Free")
        self.free3 = QtWidgets.QPushButton("Free")

        horizontalLayout = QtWidgets.QHBoxLayout()
        horizontalLayout.addWidget(self.autoLabel)
        horizontalLayout.addWidget(self.free1)
        horizontalLayout.addWidget(self.free2)
        horizontalLayout.addWidget(self.free3)

        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addWidget(self.widget)

        verticalLayout.addLayout(horizontalLayout)
        self.setLayout(verticalLayout)

        # self.setGeometry(10, 10, 1000, 600)
        self.show()

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Q:
            self.deleteLater()

        self._key_press = event.key()
        event.accept()

    def qt_get_keypress(self):
        ret = self._key_press
        if self._key_press is not None:
            print(self._key_press)
        self._key_press = None
        return ret

    def qt_connections(self):
        self.autoLabel.clicked.connect(self.on_autoLabel_clicked)
        self.free1.clicked.connect(self.on_free1button_clicked)


    def viewCurrentImage(self):
        try:
            # setCurrentImages(images)
            out = cv2.transpose(getCurrentImages().small_rgb)
            out = cv2.flip(out, flipCode=1)
            self.img.setImage(out)
            self.display_image = out


        except Exception as e:
            logger.error(e)
            pass

    def set_horizon_points(self, pos = None):
        if pos is None:
            n_points = 20
            (c,r,z) = self.display_image.shape
            x = np.linspace(1, c, n_points+1)
            pos = np.column_stack((x, np.ones(n_points+1) * r//4))
        else:
            n_points = pos.shape[0]

        adj = np.stack([np.arange(n_points), np.arange(1, n_points+1)]).transpose()
        pen = pg.mkPen('r', width=2)
        self.g.setData(pos=pos, adj=adj, size=20, pen=pen, symbol='o', symbolPen=pen, symbolBrush=(50, 50, 200, 00), pxMode=True)


    def on_autoLabel_clicked(self):
        images = getCurrentImages()
        print ("on_autoLabel_clicked")

    def on_free1button_clicked(self):
        print ("on_free1button_clicked")


import cv2

class Viewer:
    def __init__(self, process):
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.process = process
        # self.images = Images()

        self.app = QtWidgets.QApplication(sys.argv)
        self.app.setApplicationName('Viewer')
        self.imageWidget = Widget()

        self.timer.timeout.connect(self.run_timer)
        self.timer.start(1)
        # self.app.exec_()

    def open(self):
        self.app.exec_()

    def close(self):
        print('bye')
        sys.exit(self.app)

    def run_timer(self):

        go = self.process(self, self.imageWidget.qt_get_keypress)
        if go:
            self.timer.start(1)
        else:
            self.close()

    def setCurrentImages(self, g_images, with_image):
        setCurrentImages(g_images, with_image)

    def viewCurrentImage(self):

        self.imageWidget.viewCurrentImage()
        self.imageWidget.set_horizon_points()




if __name__ == '__main__':
    g_images = Images()
    # image = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)

    # images = Images()
    # image = cv2.cvtColor(cv2.imread())
    image = cv2.cvtColor(cv2.imread('/home/jn/data/Karioitahi_09Feb2022/132MSDCF-28mm-f4/DSC01013.JPG'), cv2.COLOR_RGB2BGR)
    setCurrentImages(g_images, image)

    def test(_viewer, qt_get_keypress):
        # _viewer.update_image(images)

        cv2.imshow('small_rgb', cv2.cvtColor(getCurrentImages().small_rgb, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(100)

        if k == ord('q') or k == 27:
            return False

        return True


    viewer = Viewer(test)
    viewer.viewCurrentImage()
    viewer.open()
    viewer.close()

