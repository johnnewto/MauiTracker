__all__ = ['Viewer']

import pyqtgraph as pg
import sys
from pyqtgraph.Qt import QtCore, QtWidgets
# from PyQt5 import QtCore, QtGui
import numpy as np
from utils.image_utils import *
from utils.g_images import *
from utils import parameters as pms
from utils.horizon import find_sky_2
import cv2
import csv

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

        if shiftkey:
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
                # move neighbour points by 1/2
                dragAmount = ev.pos()[1] + self.dragOffsets[ind][1] - self.startPoints[ind][1]
                self.data['pos'][ind][1] += dragAmount
                self.data['pos'][ind-1][1] += dragAmount*0.5
                self.data['pos'][ind+1][1] += dragAmount*0.5

        if ctrlkey:
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


# g_images = Images()
# g_images.small_rgb = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)
#
#
# def setCurrentImages(images, image):
#     global g_images
#     g_images = images
#     g_images.set(image)
#
# def getCurrentImages():
#     global g_images
#     return g_images

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
        self.graph = Graph()
        view = self.widget.addViewBox()
        view.addItem(self.img)
        view.addItem(self.graph)

        self.autoLabel = QtWidgets.QPushButton("Auto Horizon")
        self.saveCSV = QtWidgets.QPushButton("Save CSV")
        self.readCSV = QtWidgets.QPushButton("Read CSV")
        self.free3 = QtWidgets.QPushButton("Free")

        horizontalLayout = QtWidgets.QHBoxLayout()
        horizontalLayout.addWidget(self.autoLabel)
        horizontalLayout.addWidget(self.saveCSV)
        horizontalLayout.addWidget(self.readCSV)
        horizontalLayout.addWidget(self.free3)

        verticalLayout = QtWidgets.QVBoxLayout()
        verticalLayout.addWidget(self.widget)

        verticalLayout.addLayout(horizontalLayout)
        self.setLayout(verticalLayout)

        self.setGeometry(100, 100, 1500, 1000)
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
        self.saveCSV.clicked.connect(self.on_saveCSV_clicked)
        self.readCSV.clicked.connect(self.on_readCSV_clicked)
        self.free3.clicked.connect(self.on_free3button_clicked)

    def viewCurrentImage(self, img=None):
        try:
            if img is None:
                img = getGImages().small_rgb
            out = cv2.transpose(img)
            out = cv2.flip(out, flipCode=1)
            self.img.setImage(out)
            self.display_image = out


        except Exception as e:
            logger.error(e)
            pass

    def set_horizon_points(self, pos = None):
        if pos is None:
            n_points = pms.NUM_HORIZON_POINTS
            (c,r) = self.display_image.shape[:2]
            x = np.linspace(0, c-1, n_points)+0.5
            pos = np.column_stack((x, np.ones(n_points) * r//4))
            pass
        else:
            n_points = pos.shape[0]
            (cols, rows) = self.display_image.shape[:2]
            pos[:,0] = pos[:,0] * cols
            pos[:,1] = pos[:,1] * rows
            assert max(pos[:,0]) < cols , 'columns must be less than image width '
        adj = np.stack([np.arange(n_points-1), np.arange(1, n_points)]).transpose()
        pen = pg.mkPen('r', width=2)
        self.graph.setData(pos=pos, adj=adj, size=10, pen=pen, symbol='o', symbolPen=pen, symbolBrush=(50, 50, 200, 00), pxMode=False)


    def on_autoLabel_clicked(self):
        print("on_autoLabel_clicked Not Implemented")
        # raise NotImplementedError

    def on_saveCSV_clicked(self):
        print("on_showCSV_clicked Not Implemented")
        # raise NotImplementedError

    def on_readCSV_clicked(self):
        print ("on_readCSV_clicked Not Implemented")
        # raise NotImplementedError

    def on_free3button_clicked(self):
        print ("on_free1button_clicked Not Implemented")
        # raise NotImplementedError


import cv2

class Viewer(Widget):
    def __init__(self, process):
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.process = process
        self.app = QtWidgets.QApplication(sys.argv)
        self.app.setApplicationName('Viewer')
        self.timer.timeout.connect(self.run_timer)
        self.timer.start(1)
        super().__init__()

    def open(self):
        self.app.exec_()

    def close(self):
        print('bye')
        sys.exit(self.app)

    def run_timer(self):

        go = self.process(self, self.qt_get_keypress)
        # go = self.process(self, self.imageWidget.qt_get_keypress)
        if go:
            self.timer.start(1)
        else:
            self.close()

    def setCurrentImages(self, g_images, with_image):
        setGImages(g_images, with_image)

    def setCurrentImage(self, img=None):

        # self.imageWidget.viewCurrentImage(img)
        # self.imageWidget.set_horizon_points()
        self.viewCurrentImage(img)
        self.set_horizon_points()

    def on_autoLabel_clicked(self):
        pos = mask2pos(getGImages().horizon)
        print ("on_autoLabel_clicked")
        print(pos)
        self.set_horizon_points(pos)


    def on_saveCSV_clicked(self):
        print("on_saveCSV_clicked")
        # print(self.graph.data['pos'])
        pos = self.graph.data['pos']
        (cols, rows) = self.display_image.shape[:2]
        pos[:, 0] = pos[:, 0] / cols
        pos[:, 1] = pos[:, 1] / rows
        writecsv('filename1', pos)

    def on_readCSV_clicked(self):
        print("on_readCSV_clicked")
        pos =  readcsv('filename1')
        self.set_horizon_points(pos)

def writecsv(filename, pos):
    with open(filename, 'w') as f:
        write = csv.writer(f)
        write.writerows([[filename]])
        write.writerows(pos)


def readcsv(filename):
    with open(filename, 'r') as file:        # self.imageWidget = Widget()
        reader = csv.reader(file)
        lines = []
        for row in reader:
            lines.append(row)
            print(row)
    filename = lines[0][0]
    lines = [[float(c) for c in r] for r in lines[1:] ]
    lines = np.array(lines)
    return lines

def mask2pos(mask):
    n_points = pms.NUM_HORIZON_POINTS
    (rows, cols) = mask.shape
    c_pnts = np.linspace(0, cols - 1, n_points + 1) + 0.5
    c_pnts[-1] = cols-1
    pos = []
    for c in c_pnts:
        # find row in column c that is non zero
        vcol = mask[::-1,int(c)]
        hpnt = np.argmax(vcol==0)
        pos.append([c/cols, hpnt/rows])

    return np.asarray(pos)
    # return


from utils.horizon import *
if __name__ == '__main__':
    # g_images = Images()
    # image = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)

    # images = Images()
    # image = cv2.cvtColor(cv2.imread())
    # filename = '/home/jn/data/Tairua_15Jan2022/109MSDCF/DSC03288.JPG'
    filename = '/home/jn/data/Karioitahi_09Feb2022/132MSDCF-28mm-f4/DSC01013.JPG'
    image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_RGB2BGR)

    setGImages(image)
    getGImages().mask_sky()
    gray_img_s = getGImages().small_gray.copy()
    getGImages().horizon = set_horizon(gray_img_s)
    cv2.imshow('horizon', getGImages().horizon)

    cv2.imshow('mask_sky', getGImages().mask)

    pos = mask2pos(getGImages().horizon)

    def test(_viewer, qt_get_keypress):
        # _viewer.update_image(images)

        # cv2.imshow('small_rgb', cv2.cvtColor(getGImages().small_rgb, cv2.COLOR_RGB2BGR))
        k = cv2.waitKey(100)

        if k == ord('q') or k == 27:
            return False

        return True


    viewer = Viewer(test)
    viewer.setCurrentImage()
    # viewer.viewCurrentImage(getGImages().mask)
    viewer.open()
    viewer.close()

