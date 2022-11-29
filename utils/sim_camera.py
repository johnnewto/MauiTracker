__all__ = ['CameraThread']


import cv2
import glob
from turbojpeg import TurboJPEG, TJPF_RGB, TJPF_BGR

from typing import List
import threading, time
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def init_sim_camera(cam_name, path):
    pass

#     self.path = path
#     self.dataset = self.parse_input(self.path)
#     self.dataset.sort()
#     self.frame_num = 0
#     self._direction_fwd = True
#     self.restep = False
    # assert len(cam_serials) == len(cam_names), "len(cam_serials) must equal len(cam_names)"
    # get instance of the pylon TransportLayerFactory
    # all pypylon objects are instances of SWIG wrappers around the underlying pylon c++ types
    # if pixelformat == '12p':
    #     pixelformat = 'BayerRG12p'
    # else:
    #     pixelformat = 'BayerRG8'


    # tlf = py.TlFactory.GetInstance()
    # _devices = tlf.EnumerateDevices()

    # order devices by cam_names
    # devices = [None]*len(cam_serials)
    # for d in _devices:
    #     sn = d.GetSerialNumber()
    #     if sn in cam_serials:
    #         idx = cam_serials.index(sn)
    #         devices[idx] = d
    #
    # for name, d in zip(cam_names, devices):
    #     logging.info(f'Found camera {name}, {d.GetModelName()}, {d.GetSerialNumber()}')
    #
    # METHOD = "use pylon InstantCameraArray"
    #
    # if METHOD == "use pylon InstantCameraArray":
    #     try:
    #         cam_array = py.InstantCameraArray(len(devices))
    #         for idx, cam in enumerate(cam_array):
    #
    #             cd = tlf.CreateDevice(devices[idx])
    #             cam.Attach(cd)
    #         cam_array.Open()
    #     except:
    #         raise RuntimeError(f'Error when attaching camera {cam_names[idx]}, {devices[idx].GetSerialNumber()}')
    # else:
    #     cam_array = []
    #     for d in devices:
    #         logging.info(d.GetModelName(), d.GetSerialNumber())
    #
    #         cam = py.InstantCamera(tlf.CreateDevice(d))
    #         cam.Open()
    #         cam_array.append(cam)
    #
    # # store a unique number for each camera to identify the incoming images
    # for idx, cam in enumerate(cam_array):
    #     camera_serial = cam.DeviceInfo.GetSerialNumber()
    #     # print(f"set context {idx} for camera {camera_serial}")
    #     cam.SetCameraContext(idx)
    #
    # # set the parameters for each camera
    # for idx, cam in enumerate(cam_array):
    #     camera_serial = cam.DeviceInfo.GetSerialNumber()
    #     logging.info(f"Setting parameters for camera {camera_serial}")
    #     # cam.BalanceWhiteAuto = 'Off'   # todo need this here otherwise cam sometimes loses white balance ??
    #     cam.ExposureTime = 6000
    #     cam.AcquisitionFrameRateEnable = True
    #     cam.AcquisitionFrameRate = 10
    #     # cam.Height = 340
    #     # cam.OffsetY = 350
    #     # cam.BslColorSpace = 'Off'
    #
    #     cam.PixelFormat = pixelformat
    #     cam.Gain = 0
    #     # cam.BalanceWhiteAuto = 'Continuous' # todo need this here otherwise cam sometimes loses white balance ??
    #
    # return cam_array



class CameraThread:
    def __init__(self, name, path, mode='RGB', cvtgray=True, ):
        self.path = path
        self.mode = mode
        self.cvtgray = cvtgray

        self.dataset = self.parse_input(self.path)
        self.get_frame_num = 0
        self.frameid = -1
        self._direction_fwd = True
        self.restep = False
        self.jpeg_reader = TurboJPEG()  # create TurboJPEG object for image reading

        self.name = name
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.process_image = None
        self.cvtgray = False
        if self.mode == "RGB":
            self.pixel_format = TJPF_RGB
        elif self.mode == "BGR":
            self.pixel_format = TJPF_BGR

    def parse_input(self, path):
        files =  glob.glob(self.path + '*.JPG')
        if len(files) <= 1:
            print(f"No files found in {self.path}")
            return []
        files.sort()
        return files

    def start(self):
        self.stopped = False
        self.thread.start()

    def stop(self):
        """indicate that the thread should be stopped"""
        self.stopped = True
        # wait until stream resources are released ( thread might be still grabbing frame)
        self.thread.join()

    def close(self):
        self.stop()

    # def get_initial_time(self):
    #     for i in range(1000):
    #         with self.cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException) as res:
    #             if res and res.GrabSucceeded():
    #                 return res.TimeStamp
    #             else:
    #                 time.sleep(0.001)
    #     return None

    # def set_cam_offsetY(self, amount):
    #     self.cam.OffsetY = amount

    def read_first_image(self):
        if self.frame_num < len(self.dataset):
            with open(self.dataset[self.frame_num], "rb") as file:
                img = self.jpeg_reader.decode(file.read(), pixel_format=self.pixel_format)
                # if img.ndim == 3 and self.cvtgray:
                #     return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                return  (img, self.dataset[self.frame_num])
        else:
            logger.warning("self.frame_num > len(self.dataset)")

    def get_next(self, next_idx):
        if next_idx >= 0 and next_idx < len(self.dataset):
            self.get_frame_num = next_idx
            time.sleep(0.001)
        return self.get_frame_num

    def read(self, idx):
        # return the frame most recently read
        if self.frameid == idx:
            # frameID in memory is correct
            self.lastcapture = (True, self.image, self.frameid)
        else:
            self.get_frame_num = idx
            if self.event.wait(timeout=0.1): # 1 second timeout
                self.lastcapture = (True, self.image, self.frameid)
            else:
                self.lastcapture = (False, None, None)

        self.event.clear()
        return self.lastcapture

    def current_frameid(self):
        return self.lastcapture[2]

    def update(self):
        self.stopped = False
        # initial_time = self.get_initial_time()
        # initial_time = 0

        while not self.stopped:
            if self.frameid != self.get_frame_num:
                with open(self.dataset[self.get_frame_num], "rb") as file:
                    # del self._img
                    # gc.collect()
                    self._img = self.jpeg_reader.decode(file.read(), pixel_format=self.pixel_format)
                    if self._img.ndim == 3 and self.cvtgray:
                        self._img = cv2.cvtColor(self._img, cv2.COLOR_RGB2GRAY)
                    self.image = (self._img, self.dataset[self.get_frame_num])
                    self.frameid = self.get_frame_num
                self.event.set()
                # print('Got frame', self.get_frame_num)
            else:
                time.sleep(0.001)

        #     # if self._img.ndim == 3 and self.cvtgray:
        #     with self.cam.RetrieveResult(1000) as res:
        #         if res and res.GrabSucceeded():
        #
        #             self.img_nr = res.ImageNumber
        #             self.cam_id = res.GetCameraContext()
        #             # self.timestamp = res.TimeStamp - initial_time
        #             self.frameid = res.ID
        #             self.image = res.GetArray()
        #
        #             if self.process_image is not None:
        #                 self.process_image(self.image, self.frameid)
        #
        #             self.event.set()
        #         else:
        #             time.sleep(0.001)
        #
        # self.cam.StopGrabbing()
        # self.cam.Close()
        logging.info(f'Camera {self.name} is closed')



    def set_callback(self, process_image):
        """  Set the callback on frame capture """
        self.process_image = process_image



if __name__ == '__main__':
    from imutils import resize
    import numpy as np
    from pathlib import Path

    home = str(Path.home())

    # path = 'Z:/Day2/seq1/'
    filename = home + "/data/large_plane/images/DSC00176.JPG"
    path = home + "/data/testImages/original/"

    cv2.imshow('test', np.zeros((100, 100, 3), dtype='uint8'))

    # exit()

    # cam_serials = ['23479535']
    # cam_names = ['FrontCentre']
    # cam_serials = ['40083688']
    # cam_names = ['FrontCentre']
    # cam_offsetYs = [280]
    cam = CameraThread('simcam', path, )
    cam.start()
    idx = 1
    direction = 1
    while True:
        try:
            # for i, cam in enumerate(cameras.cam_threads):
                (grabbed, (frame, filename), id) = cam.read(idx)
                if grabbed:
                    # frame = cv2.cvtColor(framebayer, cv2.COLOR_BAYER_BG2BGR)
                    # frame = cv2.cvtColor(framebayer, cv2.COLOR_BAYER_BG2GRAY)
                    # img_scaled = frame
                    # if frame.dtype == np.uint16:
                    #     img_scaled = cv2.normalize(frame, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
                    img_scaled = resize(frame, width=1200)
                    [r,c] = frame.shape[:2]
                    # sz = 500
                    # img_scaled = img_scaled[r//2-sz:r//2+sz, c//2-sz:c//2+sz]
                    cv2.putText(img_scaled, f"{id}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    cv2.imshow(f'{cam.name}', img_scaled)
                    # cv2.imshow('horizon', np.zeros((100,100,3),dtype='uint8'))
                else:
                    pass

        except KeyboardInterrupt:
            break
        except Exception as msg:
            logger.error(msg)
        else:
            time.sleep(0.01)

        k = cv2.waitKey(1)
        if k == 27 or k == 3 or k == ord('q'):
            break  # esc to quit
        if k == ord('d'):
            direction *= -1

        if k == ord(' '):
            idx += direction
            idx = cam.get_next(idx)
            timeout = time.time() + 0.21  # seconds from now
            while True:
                if time.time() > timeout:
                    break




    cam.close()