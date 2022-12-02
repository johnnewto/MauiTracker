__all__ = ['CameraThread','init_cameras','RunCameras']

import pypylon.pylon as py

import cv2
import imutils
from imutils.video import FPS
from typing import List
import threading, time
import logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def init_cameras(cam_serials, cam_names, pixelformat='8'):
    # assert len(cam_serials) == len(cam_names), "len(cam_serials) must equal len(cam_names)"
    # get instance of the pylon TransportLayerFactory
    # all pypylon objects are instances of SWIG wrappers around the underlying pylon c++ types
    if pixelformat == '12p':
        pixelformat = 'BayerRG12p'
    else:
        pixelformat = 'BayerRG8'


    tlf = py.TlFactory.GetInstance()
    _devices = tlf.EnumerateDevices()

    # order devices by cam_names
    devices = [None]*len(cam_serials)
    for d in _devices:
        sn = d.GetSerialNumber()
        if sn in cam_serials:
            idx = cam_serials.index(sn)
            devices[idx] = d

    for name, d in zip(cam_names, devices):
        logging.info(f'Found camera {name}, {d.GetModelName()}, {d.GetSerialNumber()}')

    METHOD = "use pylon InstantCameraArray"

    if METHOD == "use pylon InstantCameraArray":
        try:
            cam_array = py.InstantCameraArray(len(devices))
            for idx, cam in enumerate(cam_array):

                cd = tlf.CreateDevice(devices[idx])
                cam.Attach(cd)
            cam_array.Open()
        except:
            raise RuntimeError(f'Error when attaching camera {cam_names[idx]}, {devices[idx].GetSerialNumber()}')
    else:
        cam_array = []
        for d in devices:
            logging.info(d.GetModelName(), d.GetSerialNumber())

            cam = py.InstantCamera(tlf.CreateDevice(d))
            cam.Open()
            cam_array.append(cam)

    # store a unique number for each camera to identify the incoming images
    for idx, cam in enumerate(cam_array):
        camera_serial = cam.DeviceInfo.GetSerialNumber()
        # print(f"set context {idx} for camera {camera_serial}")
        cam.SetCameraContext(idx)

    # set the parameters for each camera
    for idx, cam in enumerate(cam_array):
        camera_serial = cam.DeviceInfo.GetSerialNumber()
        logging.info(f"Setting parameters for camera {camera_serial}")
        # cam.BalanceWhiteAuto = 'Off'   # todo need this here otherwise cam sometimes loses white balance ??
        cam.ExposureTime = 6000
        cam.AcquisitionFrameRateEnable = True
        cam.AcquisitionFrameRate = 10
        # cam.Height = 340
        # cam.OffsetY = 350
        # cam.BslColorSpace = 'Off'

        cam.PixelFormat = pixelformat
        cam.Gain = 0
        # cam.BalanceWhiteAuto = 'Continuous' # todo need this here otherwise cam sometimes loses white balance ??

    return cam_array



class CameraThread:
    def __init__(self, name, cam):
        self.cam = cam
        self.name = name
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.process_image = None
        self.frame_num = -1

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

    def get_initial_time(self):
        for i in range(1000):
            with self.cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException) as res:
                if res and res.GrabSucceeded():
                    return res.TimeStamp
                else:
                    time.sleep(0.001)
        return None

    def set_cam_offsetY(self, amount):
        self.cam.OffsetY = amount

    def update(self):
        self.stopped = False
        initial_time = self.get_initial_time()
        # initial_time = 0

        while not self.stopped:
            with self.cam.RetrieveResult(1000) as res:
                if res and res.GrabSucceeded():

                    self.frame_num += 1
                    self.img_nr = res.ImageNumber
                    self.cam_id = res.GetCameraContext()
                    self.timestamp = res.TimeStamp - initial_time
                    self.frameid = res.ID
                    self.image = res.GetArray()

                    if self.process_image is not None:
                        self.process_image(self.image, self.frameid)

                    self.event.set()
                else:
                    time.sleep(0.001)

        self.cam.StopGrabbing()
        self.cam.Close()
        logging.info(f'Camera {self.name} is closed')

    def read(self):
        # return the frame most recently read
        if self.event.wait(timeout=0.1): # 1 second timeout
            self.lastcapture = (True, self.image, self.frameid)
        else:
            self.lastcapture = (False, None, None)

        self.event.clear()
        return self.lastcapture

    def set_callback(self, process_image):
        """  Set the callback on frame capture """
        self.process_image = process_image

    def __iter__(self):
        return self

    def __next__(self):
        grabbed = False
        if self.event.wait(timeout=0.5): # 0.1 second timeout
            try:
                self.fps.update()
            except AttributeError:
                self.fps = FPS().start()
            self.event.clear()
            grabbed = True

        return (self.image, "basler"), self.frame_num, grabbed


class RunCameras:

    def __init__(self, cam_names, cam_serials,  pixelformat='8', dostart=False):
        # cam_serials = ['40083688', '40072295', '40072285']
        # cam_names = ['FrontCentre', 'FrontRight', 'FrontLeft']
        # cam_offsetYs = [280, 350, 350]
        self.cam_array = init_cameras(cam_serials, cam_names, pixelformat=pixelformat)

        self.cam_threads = []
        for name, cam in zip(cam_names, self.cam_array):
            self.cam_threads.append(CameraThread(name, cam))

        self.cam_array.StartGrabbing()

        # for thd, offsetY in zip(self.cam_threads, cam_offsetYs):
        #     thd.set_cam_offsetY(offsetY)

        if dostart:
            for thd in self.cam_threads:
                thd.start()

    def close(self):
        # thd: CameraThread
        for thd in self.cam_threads:
            thd.stop()

    def get_camera(self, name) -> CameraThread:
        for thd in self.cam_threads:
            if name == thd.name:
                return thd
        raise RuntimeError(f'{name} is not found in camera threads')


if __name__ == '__main__':
    from imutils import resize
    import numpy as np

    cv2.imshow('test', np.zeros((100, 100, 3), dtype='uint8'))

    # exit()

    cam_serials = ['23479535']
    cam_names = ['FrontCentre']
    # cam_serials = ['40083688']
    # cam_names = ['FrontCentre']

    cameras = RunCameras(cam_names, cam_serials,  pixelformat='12p')

    # test get_camera
    cam1 = cameras.get_camera('FrontCentre')
    assert cam1.name == 'FrontCentre'

    for thd in cameras.cam_threads:
        thd.start()
        # cv2.namedWindow(f'{thd.name}', cv2.WINDOW_NORMAL)

    # while True:
    #     try:
    for (image, filename), frameNum, grabbed in iter(cam1):
        framebayer = image

        # for i, cam in enumerate(cameras.cam_threads):
        #     (grabbed, framebayer, id) = cam.read()
        try:
            if grabbed:
                # frame = cv2.cvtColor(framebayer, cv2.COLOR_BAYER_BG2BGR)
                frame = cv2.cvtColor(framebayer,  cv2.COLOR_BAYER_BG2BGR)
                img_scaled = imutils.resize(frame, width=1000)
                if frame.dtype == np.uint16:
                    img_scaled = cv2.normalize(img_scaled, dst=None, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
                # frame = resize(frame, width=1200)
                [r,c] = frame.shape[:2]
                # sz = 500
                # img_scaled = img_scaled[r//2-sz:r//2+sz, c//2-sz:c//2+sz]
                cv2.putText(img_scaled, f"{frameNum}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow(f'{cam1.name}', img_scaled)
                print(frameNum)
                # k = cv2.waitKey(1)
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

    cameras.close()