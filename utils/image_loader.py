import gc
import os
import threading
from abc import abstractmethod
from timeit import default_timer as timer
import time
import cv2
import numpy as np
from PIL import Image
import glob
from imutils import resize
from imutils.video import FPS
from pympler import muppy

from turbojpeg import TurboJPEG, TJPF_GRAY, TJPF_RGB, TJPF_BGR,  TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT

from motrackers import parameters as pms
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)


class _ImageLoader:
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",)

    def __init__(self, path: str, mode: str = "BGR"):
        self.open_path(path)
        self.mode = mode

    def open_path(self, path):
        self.path = path
        self.dataset = self.parse_input(self.path)
        self.dataset.sort()
        self.frame_num = 0
        self.direction_fwd = True
        self.restep = False

    def parse_input(self, path):
        files =  glob.glob(self.path)
        if len(files) <= 1:
            print(f"No files found in {self.path}")
            return []
        return files

        # # single image or tfrecords file
        # if os.path.isfile(path):
        #     assert path.lower().endswith(
        #         self.extensions,
        #     ), f"Unsupportable extension, please, use one of {self.extensions}"
        #     return [path]

        # if os.path.isdir(path):
        #     # lmdb environment
        #     if any([file.endswith(".mdb") for file in os.listdir(path)]):
        #         return path
        #     else:
        #         # folder with images
        #         paths = [os.path.join(path, image) for image in os.listdir(path)]
        #         return paths





    def __iter__(self):
        # self.frame_num = 0
        return self

    def __len__(self):
        return len(self.dataset)

    @abstractmethod
    def __next__(self):
        pass


class CV2Loader(_ImageLoader):
    def __next__(self):
        start = timer()
        path = self.dataset[self.frame_num]  # get image path by index from the dataset
        image = cv2.imread(path)  # read the image
        full_time = timer() - start
        if self.mode == "RGB":
            start = timer()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change color mode
            full_time += timer() - start
        self.frame_num += 1
        return image, full_time


class PILLoader(_ImageLoader):
    def __next__(self):
        start = timer()
        path = self.dataset[self.frame_num]  # get image path by index from the dataset
        image = np.asarray(Image.open(path))  # read the image as numpy array
        full_time = timer() - start
        if self.mode == "BGR":
            start = timer()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # change color mode
            full_time += timer() - start
        self.frame_num += 1
        return image, full_time

# class old_TurboJpegLoader(ImageLoader):
#     def __init__(self, path, **kwargs):
#         super(TurboJpegLoader, self).__init__(path)
#         self.jpeg_reader = TurboJPEG()  # create TurboJPEG object for image reading
#         self.kwargs = kwargs
#         self.img_cnt = 0
#         self.lock = threading.Lock()
#         if 'mode' in kwargs.keys():
#             self.mode = kwargs['mode']
#
#     def _read_next_image(self, i):
#         print(f"read_next_image {i}")
#         time.sleep(0.01)
#
#
#     def __next__(self):
#         start = timer()
#         file = open(self.dataset[self.sample_idx], "rb")  # open the input file as bytes
#
#         full_time = timer() - start
#         if self.mode == "RGB":
#             pixel_format = TJPF_RGB
#         elif self.mode == "BGR":
#             pixel_format = TJPF_BGR
#         start = timer()
#         image = self.jpeg_reader.decode(file.read(), pixel_format=pixel_format)  # decode raw image
#         full_time += timer() - start
#         self.sample_idx += 1
#         t = threading.Thread(target=self._read_next_image, args=(self.img_cnt,))
#         t.start()
#         self.img_cnt += 1
#         return image, full_time
#
INTHREAD = True
class ImageLoader(_ImageLoader):
    def __init__(self, path, mode='RGB', cvtgray=True, start_frame=None, **kwargs):
        super(ImageLoader, self).__init__(path)
        self.jpeg_reader = TurboJPEG()  # create TurboJPEG object for image reading
        self.mode = mode
        self.cvtgray = cvtgray

        if start_frame is None:
            self.frame_num = 0
        else:
            self.frame_num = start_frame

        self.kwargs = kwargs
        self.select = 0
        self.lock = threading.Lock()
        if 'mode' in kwargs.keys():
            self.mode = kwargs['mode']
        if self.mode == "RGB":
            self.pixel_format = TJPF_RGB
        elif self.mode == "BGR":
            self.pixel_format = TJPF_BGR
        self.firstimage = self.read_first_image()
        self.image = self.firstimage
        self.fps = FPS().start()
        self._img = None

    def get_FPS(self):
        """ the the FPS of the calls """
        self.fps.stop()
        _fps = self.fps.fps()
        self.fps.start()
        self.fps._numFrames = 0
        return _fps

    def read_first_image(self):
        if self.frame_num < len(self.dataset):
            with open(self.dataset[self.frame_num], "rb") as file:
                img = self.jpeg_reader.decode(file.read(), pixel_format=self.pixel_format)
                if img.ndim == 3 and self.cvtgray:
                    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                return  (img, self.dataset[self.frame_num])
        else:
            logger.warning("self.frame_num > len(self.dataset)")

    def _read_next_image(self, idx, sel):
        """ this is run in a thread"""
        with open(self.dataset[idx], "rb")  as file:
            # del self._img
            # gc.collect()
            self._img = self.jpeg_reader.decode(file.read(), pixel_format=self.pixel_format)
            if self._img.ndim == 3 and self.cvtgray:
                self._img = cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY)

            # wait until last image is processed (release the lock to allow the new image)
            self.lock.acquire()
            self.image = (self._img, self.dataset[idx])
            self.lock.release()

    def _release_last_image(self):
        if self.lock.locked():
            self.lock.release()  # this will allow last image to be released

    def __next__(self):
        self.fps.update()
        if not self.restep:
            self.frame_num += 1 if self.direction_fwd else -1
        self.restep = False

        # self.sample_idx += 1
        if self.frame_num < self.__len__() and self.frame_num >= 0:
            if INTHREAD:
                self._release_last_image()
                # running in separate thread allows the jpeg loading to take place during any sleep/wait time
                self.t = threading.Thread(target=self._read_next_image, args=(self.frame_num, self.select))
                self.t.start()

                self.lock.acquire()
            else:
                self._read_next_image(self.frame_num, self.select)
            return self.image, self.frame_num, True

        else:
            raise StopIteration


    def close(self):
        if self.lock.locked():
            self.lock.release()


def getloader(method, path, **kwargs):
    return methods[method](path, **kwargs)  # get the image loader

methods = {
    "cv2": CV2Loader,
    "pil": PILLoader,
    "turbojpeg": ImageLoader,
}


if __name__ == '__main__':
    from utils.show_images import putText
    from pathlib import Path

    home = str(Path.home())

    # path = 'Z:/Day2/seq1/'
    filename = home + "/data/large_plane/images/DSC00176.JPG"
    path = home + "/data/large_plane/images/"


    loader = ImageLoader(path + '*.JPG', mode='BGR', cvtgray=False)

    wait_timeout = 30     # todo !!! need to fix missing frames, might need to try a queue
    last_img_path = ''
    # (image, img_path), frameNum, grabbed
    for (image, img_path), frameNum, grabbed in iter(loader):
        # cmo =  update(cmo)
        # img = next(images)
        # print(f"processing {idx}, {timer()}")
        # time.sleep(0.01)
        # # print("")
        # continue
        if img_path == last_img_path:
            time.sleep(0.001)
            continue

        last_img_path = img_path
        image = resize(image, width=500)
        putText(image, f'Frame = {frameNum}, {timer() :.3f}')
        print(f'Frame = {frameNum}, {img_path}')
        # continue
        cv2.imshow('image',  image)
        k = cv2.waitKey(wait_timeout)
        if k == ord('q') or k == 27:
            break
        if k == ord(' '):
            wait_timeout = 0
        if k == ord('d'):
            wait_timeout = 0
            loader.direction_fwd = not loader.direction_fwd
        if k == ord('g'):
            wait_timeout = 100
        if k == ord('r'):
            # change direction
            wait_timeout = 0
            loader.restep = True

    print(f'FPS = {loader.get_FPS()}')

    loader.close()

