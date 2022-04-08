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

from turbojpeg import TurboJPEG, TJPF_GRAY, TJPF_RGB, TJPF_BGR,  TJSAMP_GRAY, TJFLAG_PROGRESSIVE, TJFLAG_FASTUPSAMPLE, TJFLAG_FASTDCT


class _ImageLoader:
    extensions: tuple = (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif",)

    def __init__(self, path: str, mode: str = "BGR"):
        self.path = path
        self.mode = mode
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

        if os.path.isdir(path):
            # lmdb environment
            if any([file.endswith(".mdb") for file in os.listdir(path)]):
                return path
            else:
                # folder with images
                paths = [os.path.join(path, image) for image in os.listdir(path)]
                return paths





    def __iter__(self):
        self.frame_num = 0
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

class ImageLoader(_ImageLoader):
    def __init__(self, path, mode='RGB', cvtgray=True, **kwargs):
        super(ImageLoader, self).__init__(path)
        self.jpeg_reader = TurboJPEG()  # create TurboJPEG object for image reading
        self.mode = mode
        self.cvtgray = cvtgray
        self.kwargs = kwargs
        self.frame_num = 0
        self.select = 0
        self.lock = threading.Lock()
        if 'mode' in kwargs.keys():
            self.mode = kwargs['mode']
        if self.mode == "RGB":
            self.pixel_format = TJPF_RGB
        elif self.mode == "BGR":
            self.pixel_format = TJPF_BGR

        self._img = self.read_first_image()
        self.get_next_event = threading.Event()
        self.got_next_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        self.fps = FPS().start()
        self.firstimage = self.__next__()
        self.image = [0, self.firstimage]




    def get_FPS(self):
        """ the the FPS of the calls """
        self.fps.stop()
        _fps = self.fps.fps()
        self.fps.start()
        self.fps._numFrames = 0
        return _fps

    def read_first_image(self):
        with open(self.dataset[0], "rb") as file:
            img = self.jpeg_reader.decode(file.read(), pixel_format=self.pixel_format)
            if img.ndim == 3 and self.cvtgray:
                return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            return  img


    def _start_next_image(self):
        if self.lock.locked():
            self.lock.release()  # this will allow last image to be released
        self.get_next_event.set()


    def run(self):
        while (True):
            self.get_next_event.wait(2)
            self.get_next_event.clear()
            # time.sleep(0.0001)
            with open(self.dataset[self.frame_num], "rb") as file:
                img = self.jpeg_reader.decode(file.read(), pixel_format=self.pixel_format)
                if img.ndim == 3 and self.cvtgray:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                self.image = [self.frame_num, img]
                self.got_next_event.set()




    def __next__(self):
        self.fps.update()
        if not self.restep:
            self.frame_num += 1 if self.direction_fwd else -1
        self.restep = False

        # self.sample_idx += 1
        if self.frame_num < self.__len__() and self.frame_num >= 0:
            self.get_next_event.set()  # start decoding next image
            # print('get_next_event', time.time())
            self.got_next_event.wait(2)
            # print('got_next_event', time.time())
            self.got_next_event.clear()

            return self.image

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
    filename = home + "/data/large_plane/DSC00176.JPG"
    path = home + "/data/large_plane/"


    loader = ImageLoader(path + '*.JPG', mode='BGR', cvtgray=False)

    wait_timeout = 1

    for i, (idx, img) in enumerate(iter(loader)):
        # cmo =  update(cmo)
        # img = next(images)
        print(f"processing {idx}, {timer()}")
        # time.sleep(0.01)
        # # print("")
        img = resize(img, width=500)
        putText(img, f'Frame = {idx}, fontScale=0.5')
        cv2.imshow('image',  img)
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

    loader.close()
    print(f'FPS = {loader.get_FPS()}')
