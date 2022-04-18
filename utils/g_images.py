__all__ = ['setGImages', 'getGImages', 'Images']

import cv2 as cv2
import numpy as np
from imutils import resize
from utils import parameters as pms
from utils.horizon import *
from utils.image_utils import min_pool, BH_op
import time, sys
from dataclasses import dataclass

if sys.platform == "win32":
    # On Windows, the best timer is time.clock()
    default_timer = time.perf_counter
else:
    # On most other platforms the best timer is time.time()
    default_timer = time.time

import logging
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).parent.parent

@dataclass
class Images:
    maxpool:int = 12
    CMO_kernalsize = 3
    full_rgb:np.array = None
    small_rgb:np.array = None
    full_gray:np.array = None
    small_gray:np.array = None
    minpool:np.array = None
    minpool_f:np.array = None
    last_minpool_f:np.array = None
    cmo:np.array = None
    mask:np.array = None
    horizon:np.array = None
    file_path = None

    def set(self, image:np.array, _file_path:str=''):
        self.full_rgb = image
        self.file_path = _file_path
        if self.full_rgb.ndim == 3:
            # use this as much faster than cv2.cvtColor(imgrgb, cv2.COLOR_BGR2GRAY) (~24msec for 6K image)
            self.full_gray = self.full_rgb[:,:,1]

        self.minpool = min_pool(self.full_gray, self.maxpool, self.maxpool)
        small_gray = resize(self.full_gray, width=self.minpool.shape[1])
        self.small_rgb = resize(self.full_rgb, width=self.minpool.shape[1])
        self.small_gray = np.zeros_like(self.minpool, dtype='uint8')
        n_rows = min(self.minpool.shape[0], small_gray.shape[0])
        self.small_gray[:n_rows,:] = small_gray[:n_rows,:]
        # self.small_gray = small_gray
        self.minpool_f = np.float32(self.minpool )

    def mask_sky(self):
        self.mask = find_sky_2(self.minpool, threshold=80,  kernal_size=7)
        self.cmo = BH_op(self.minpool, (self.CMO_kernalsize, self.CMO_kernalsize))
        self.cmo[self.mask > 0] = 0



# Set global images buffer
g_images = Images()
g_images.small_rgb = np.random.normal(size=(320, 480, 3), loc=1024, scale=64).astype(np.uint16)

def setGImages(image, file_path=None):
    global g_images
    g_images.set(image, file_path)

def getGImages() -> Images:
    global g_images
    return g_images


if __name__ == '__main__':
    from utils.show_images import putText
    #
    # (rows, cols) = (2000, 3000)
    # center = (2750, 4350)
    # (_r, _c) = (center[0]-rows//2, center[1]-cols//2)
    # crop = [_r, _r + rows, _c, _c + cols]
    # home = str(Path.home())
    # images = ImageLoader(home+"/data/large_plane/images.npy", crop=crop, scale=0.1, color='Gray')
    # wait_timeout = 100
    # for img, i in images:
    #     # cmo =  update(cmo)
    #     # img = next(images)
    #     img = resize(img, width=500)
    #     putText(img, f'Frame = {i}, fontScale=0.5')
    #     cv2.imshow('image',  img)
    #     k = cv2.waitKey(wait_timeout)
    #     if k == ord('q') or k == 27:
    #         break
    #     if k == ord(' '):
    #         wait_timeout = 0
    #     if k == ord('d'):
    #         wait_timeout = 0
    #         images.direction_fwd = not images.direction_fwd
    #     if k == ord('g'):
    #         wait_timeout = 100
    #     if k == ord('r'):
    #         # change direction
    #         wait_timeout = 0
    #         images.restep = True
    #
    # cv2.waitKey(1000)
    # cv2.destroyAllWindows()

