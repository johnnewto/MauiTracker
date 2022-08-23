import math
import time
import cv2
from pathlib import Path
import numpy as np

from utils.g_images import *
from utils.qt_viewer import *

from utils.image_utils import *
from utils.horizon import *
from utils.show_images import *
# import utils.image_loader as il
from utils.sim_camera import CameraThread

import utils.qt_parameter_tree as ptree
from pyqtgraph.Qt import QtCore
from utils import parameters as pms
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)


# def mask2pos(mask):
#     n_points = pms.NUM_HORIZON_POINTS
#     (rows, cols) = mask.shape
#     c_pnts = np.linspace(0, cols - 1, n_points + 1)
#     c_pnts[-1] = cols - 1
#     pos = []
#     for c in c_pnts:
#         # find row in column c that is non zero
#         vcol = mask[::-1, int(c)]
#         hpnt = np.argmax(vcol == 0)
#         pos.append([c / cols, hpnt / rows])
#
#     pos.append([c / cols, 0.0])
#     pos.append([0.0, 0.0])
#     return np.asarray(pos)
#

def btn_next_clicked(val):
    if loader.direction_fwd != True:
        get_image()
        time.sleep(1)
        loader.direction_fwd = True
    if get_image():
        image = getGImages().full_rgb
        file_path = getGImages().file_path
        viewer.viewCurrentImage(image)
        if not viewer.read_rois():
            # setTimeout(PAUSE_TIMEOUT)
            print(f'No ROI file found for {file_path}')
        frameNum = "???"
        viewer.setWindowTitle(f'File = {file_path}')

def btn_back_clicked(val):
    if loader.direction_fwd != False:
        get_image()
        time.sleep(1)
        loader.direction_fwd = False
    if get_image():
        image = getGImages().full_rgb
        file_path = getGImages().file_path
        viewer.viewCurrentImage(image)
        if not viewer.read_rois():
            # setTimeout(PAUSE_TIMEOUT)
            print(f'No ROI file found for {file_path}')
        frameNum = "???"
        viewer.setWindowTitle(f'File = {file_path}')

def old_get_image():
    try:
        (image, file_path), frameNum, grabbed = next(iter(loader))
    except Exception as e:
        logger.error(e)
        grabbed = False
        # setTimeout(PAUSE_TIMEOUT)
    if grabbed:

        if viewer.cbx_saveROIs.checkState() and viewer.getDataChanged():
            logger.info('Saving Mask')
            viewer.saveTrainMask()
            logger.info('Saving ROIS')
            viewer.saveROIS()

        setGImages(image, file_path)
        getGImages().mask_sky()
        gray_img_s = getGImages().small_gray.copy()
        getGImages().horizon = set_horizon(gray_img_s)

        # cv2.imshow('small_rgb', cv2.cvtColor(getGImages().small_rgb, cv2.COLOR_RGB2BGR))
        # cv2.imshow('mask_sky', getGImages().mask)
        # cv2.imshow('horizon', getGImages().horizon)
    return grabbed

def get_image(cam, idx):
    try:
        # (grabbed, frame, id) = cam.read(idx)
        # (image, file_path), frameNum, grabbed = next(iter(loader))
        (grabbed, (frame, filename), id) = cam.read(idx)
    except Exception as e:
        logger.error(e)
        grabbed = False
        # setTimeout(PAUSE_TIMEOUT)
    if grabbed:

        if viewer.cbx_saveROIs.checkState() and viewer.getDataChanged():
            # logger.info('Saving Mask')
            # viewer.saveTrainMask()
            logger.info('Saving ROIS')
            viewer.saveROIS()

        setGImages(frame, filename)
        getGImages().mask_sky()
        gray_img_s = getGImages().small_gray.copy()
        getGImages().horizon = set_horizon(gray_img_s)

        # cv2.imshow('small_rgb', cv2.cvtColor(getGImages().small_rgb, cv2.COLOR_RGB2BGR))
        # cv2.imshow('mask_sky', getGImages().mask)
        # cv2.imshow('horizon', getGImages().horizon)
    return grabbed

if __name__ == '__main__':

    STORE_TILES = False
    STORE_LABELS = False

    RECORD = False
    STOP_FRAME = None

    CONFIDENCE_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.2
    DRAW_BOUNDING_BOXES = True

    # gen_cmo = GenCMO(shape=(600,1000), dt=0.1, n=5)
    (rows, cols) = (2000, 3000)
    center = (2750, 4350)
    (_r, _c) = (center[0] - rows // 2, center[1] - cols // 2)
    crop = [_r, _r + rows, _c, _c + cols]
    # crop = None
    home = str(Path.home())
    # path = 'Z:/Day2/seq1/'
    path = home+"/data/large_plane/images"
    # path = home+"/data/ardmore_30Nov21"
    path = home+"/data/Karioitahi_15Jan2022/123MSDCF-35mm"
    # path = home+"/data/Karioitahi_15Jan2022/117MSDCF-28mm(subset)"
    # path = home+"/data/Karioitahi_15Jan2022/117MSDCF-28mm"
    path = home+"/data/Karioitahi_09Feb2022/132MSDCF-28mm-f4"
    # path = home+"/data/Tairua_15Jan2022/109MSDCF"
    path = home+"/data/testImages/original"
    path = home+"/data/test1"

    cam = CameraThread('simcam', path + '/*.JPG', mode='RGB')
    cam.start()
    # path = easygui.diropenbox( default=home+"/data/")

    PAUSE_TIMEOUT = 99

    def ops(viewer, qt_get_keypress):
        try:
            ops.wait_timeout += 0
            firstRun = False
        except AttributeError:
            # on first run
            ops.wait_timeout = PAUSE_TIMEOUT
            ops.direction_fwd = 1
            ops.frame_idx = 0
            firstRun = True

        def setTimeout(t):
            ops.wait_timeout = t

        def getTimeout():
            return ops.wait_timeout



        # if k == ord(' '):
        #     idx += ops.direction_fwd
        #     idx = cam.get_next(idx)
        #     timeout = time.time() + 0.21  # seconds from now
        #     while True:
        #         if time.time() > timeout:
        #             break
        k = cv2.waitKey(ops.wait_timeout)
        if k == -1:
            k = qt_get_keypress()
        if isinstance(k, str):
            k = k.upper()

        if k == QtCore.Qt.Key_Right:
            if ops.direction_fwd != 1:
                ops.direction_fwd = 1
                ops.frame_idx = cam.current_frameid() + ops.direction_fwd
                ops.frame_idx = cam.get_next(ops.frame_idx)
                time.sleep(0.2)
            k = ord(' ')

        if k == QtCore.Qt.Key_Left:
            if ops.direction_fwd != -1:
                ops.direction_fwd = -1
                ops.frame_idx = cam.current_frameid() + ops.direction_fwd
                ops.frame_idx = cam.get_next(ops.frame_idx)
                time.sleep(0.2)
            k = ord(' ')

        if ops.wait_timeout != PAUSE_TIMEOUT or k == ord(' ') or firstRun:

            if get_image(cam, ops.frame_idx):
                # print('display frame', ops.frame_idx)
                image = getGImages().full_rgb
                file_path = getGImages().file_path
                viewer.viewCurrentImage(image)
                if not viewer.read_rois():
                    setTimeout(PAUSE_TIMEOUT)

                frameNum = "???"
                viewer.setWindowTitle(f'{file_path}')
                viewer.setImageLabel(f'{file_path}')
                # viewer.setText
                # pg.TextItem(text='', color=(200, 200, 200), html=None, anchor=(0, 0), border=None, fill=None,
                #                    angle=0, rotateAxis=None)

        if k == ord('g') or k == ord('G'):
            setTimeout(10)

        if k == ord('d') or k == ord('D'):
            # change direction
            # setTimeout(PAUSE_TIMEOUT)
            ops.direction_fwd *= -1

        if k == ord('r'):
            # restep
            setTimeout(PAUSE_TIMEOUT)
            ops.restep = True

        if k is not None and k != -1:
            ops.frame_idx = cam.current_frameid() + ops.direction_fwd
            ops.frame_idx = cam.get_next(ops.frame_idx)
            # print('cam.get_next(ops.frame_idx)', ops.frame_idx)

        if k == ord('q') or k == ord('Q') or k == 27:
            # cv2.destroyAllWindows()
            # cam.close()
            return False

        return True

    # def btn_autoLabel_clicked(self):
    #     self.setDataChanged(True)
    #     pos = mask2pos(getGImages().horizon)
    #     self.set_horizon_points(pos)


    viewer = Viewer(ops)
    viewer.btn_next.clicked.connect(btn_next_clicked)
    viewer.btn_back.clicked.connect(btn_back_clicked)
    params = ptree.ParamTree()
    # params.show()
    # viewer.btn_autoLabel_clicked = btn_autoLabel_clicked

    params.win.closed.connect(viewer.close)

    # need to sub class viewer
    # viewer.win.closed.connect(viewer.close)

    # viewer.open(params)
    viewer.open(param_tree=None)

    cv2.destroyAllWindows()
    cam.close()
    # print(f'FPS = {loader.get_FPS()}')

    # if viewer is not None:
    #     viewer.close()