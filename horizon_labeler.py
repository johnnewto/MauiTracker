import math
import time
from cv2 import cv2
from pathlib import Path
import numpy as np

from utils.g_images import *
from utils.qt_viewer import *

from utils.image_utils import *
from utils.horizon import *
from utils.show_images import *
import utils.image_loader as il

from utils import parameters as pms
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)



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
    path = home+"/data/testImages"

    # path = easygui.diropenbox( default=home+"/data/")

    loader = il.ImageLoader(path + '/*.JPG', mode='RGB', cvtgray=False, start_frame=0)
    PAUSE_TIMEOUT = 99

    def ops(viewer, qt_get_keypress):
        try:
            ops.wait_timeout += 0
            firstRun = False
        except AttributeError:
            # on first run
            ops.wait_timeout = PAUSE_TIMEOUT
            firstRun = True

        def setTimeout(t):
            ops.wait_timeout = t

        def getTimeout():
            return ops.wait_timeout

        k = cv2.waitKey(ops.wait_timeout)
        if k == -1:
            k = qt_get_keypress()
        if isinstance(k, str):
            k = k.upper()
        if k == ord('q') or k == ord('Q') or k == 27:
            cv2.destroyAllWindows()
            loader.close()
            return False

        if ops.wait_timeout != PAUSE_TIMEOUT or k == ord(' ') or firstRun:
            try:
                (image, file_path), frameNum, grabbed  = next(iter(loader))
            except Exception as e:
                logger.error(e)
                grabbed = False
                setTimeout(PAUSE_TIMEOUT)
            if grabbed:
                # print(f"frame {frameNum} : {filename}  {grabbed}")
                if viewer.getDataChanged():
                    logger.info('Saving CSV')
                    viewer.saveCSV()

                if viewer.cbx_saveTrainMask.checkState():
                    logger.info('Saving Mask')
                    viewer.saveTrainMask()

                setGImages(image, file_path )
                getGImages().mask_sky()
                gray_img_s = getGImages().small_gray.copy()
                getGImages().horizon = set_horizon(gray_img_s)

                # cv2.imshow('small_rgb', cv2.cvtColor(getGImages().small_rgb, cv2.COLOR_RGB2BGR))
                cv2.imshow('mask_sky', getGImages().mask)
                cv2.imshow('horizon', getGImages().horizon)

                viewer.viewCurrentImage()
                if not viewer.readCSV():
                    setTimeout(PAUSE_TIMEOUT)
                    print(f'No csv file found for {file_path}')

                viewer.setWindowTitle(f'Frame# = {frameNum}, {file_path}')

        if k == ord('g') or k == ord('G'):
            setTimeout(1)

        if k == ord('d') or k == ord('D'):
            # change direction
            setTimeout(PAUSE_TIMEOUT)
            loader.direction_fwd = not loader.direction_fwd
        if k == ord('r'):
            # change direction
            setTimeout(PAUSE_TIMEOUT)
            loader.restep = True


        return True

    viewer = Viewer(ops).open()
    cv2.destroyAllWindows()
    loader.close()
    print(f'FPS = {loader.get_FPS()}')

    if viewer is not None:
        viewer.close()