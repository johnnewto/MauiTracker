__all__ = ['CameraThread','init_cameras','RunCameras']

import cv2

import gphoto2 as gp
from imutils.video import FPS
from turbojpeg import TurboJPEG, TJPF_RGB
from utils.image_utils import *

import threading, time
from utils import parameters as pms
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

def init_cameras(cam_serials, cam_names):
    # assert len(cam_serials) == len(cam_names), "len(cam_serials) must equal len(cam_names)"
    # get instance of the pylon TransportLayerFactory
    # all pypylon objects are instances of SWIG wrappers around the underlying pylon c++ types

    # order devices by cam_names
    devices = [None]*len(cam_serials)

    camera = gp.Camera()
    camera.init()
    cam_array = [camera]
    return cam_array



class CameraThread:
    def __init__(self, name, cam):
        self.cam = cam
        self.name = name
        self.event = threading.Event()
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.process_image = None
        self.jpeg_reader = TurboJPEG()
        self.frame_num = -1
        root = get_project_root()
        filename = root / 'experiments/images/largeplane.JPG'
        imgrgb = cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)
        # self.firstimage = (np.zeros((4000,6000,3), 'uint8'), 'nullimage')
        self.firstimage = (imgrgb, 'nullimage')
        self.image = imgrgb
        cv2.putText(self.firstimage[0], "Null Image", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)

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
        # for i in range(1000):
        #     with self.cam.RetrieveResult(1000, py.TimeoutHandling_ThrowException) as res:
        #         if res and res.GrabSucceeded():
        #             return res.TimeStamp
        #         else:
        #             time.sleep(0.001)
        return None

    def set_cam_offsetY(self, amount):
        self.cam.OffsetY = amount

    def update(self):
        self.stopped = False
        initial_time = self.get_initial_time()
        # initial_time = 0

        while not self.stopped:
            event_type, event_data = self.cam.wait_for_event(1000)
            if event_type == gp.GP_EVENT_FILE_ADDED:

                self.frame_num += 1
                cam_file = self.cam.file_get(
                    event_data.folder, event_data.name, gp.GP_FILE_TYPE_NORMAL)

                file_data = gp.check_result(gp.gp_file_get_data_and_size(cam_file))
                self.image = self.jpeg_reader.decode(file_data, pixel_format=TJPF_RGB)

                if self.process_image is not None:
                    self.process_image(self.image, self.frame_num)

                self.event.set()
            else:
                time.sleep(0.001)

        logging.info(f'Camera {self.name} is closed')

    def read(self):
        # return the frame most recently read
        if self.event.wait(timeout=0.1): # 1 second timeout
            self.lastcapture = (True, self.image, self.frame_num)
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

        return (self.image, "sony"), self.frame_num, grabbed


    def get_FPS(self):
        """ the the FPS of the calls """
        return 0
        # self.fps.stop()
        # _fps = self.fps.fps()
        # self.fps.start()
        # self.fps._numFrames = 0
        # return _fps


    def hi(self):
        return 1


class RunCameras:

    def __init__(self, cam_names, cam_serials, dostart=False):
        # cam_serials = ['40083688', '40072295', '40072285']
        # cam_names = ['FrontCentre', 'FrontRight', 'FrontLeft']
        # cam_offsetYs = [280, 350, 350]
        self.cam_array = init_cameras(cam_serials, cam_names)


        self.cam_threads = []
        for name, cam in zip(cam_names, self.cam_array):
            self.cam_threads.append(CameraThread(name, cam))

        if dostart:
            for thd in self.cam_threads:
                thd.start()
                logger.info(f'Started Camera {thd.name}')

    def close(self):
        for thd in self.cam_threads:
            thd.stop()

    def get_camera(self, name) -> CameraThread:
        for thd in self.cam_threads:
            if name == thd.name:
                return thd
        raise RuntimeError(f'{name} is not found in camera threads')


if __name__ == '__main__':
    cam_serials = ['00001']
    cam_names = ['FrontCentre']

    cameras = RunCameras(cam_names, cam_serials, dostart=True)

    # test get_camera
    cam1 = cameras.get_camera('FrontCentre')
    assert cam1.name == 'FrontCentre'
    cv2.namedWindow(f'{cam1.name}', cv2.WINDOW_NORMAL)

    for (image, filename), frameNum, grabbed in iter(cam1):
        # if frameNum > -1:
        cv2.putText(image, f"{frameNum} {grabbed} ", (100, 205), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
        cv2.imshow(f'{cam1.name}', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        k = cv2.waitKey(1)
        if k == 27 or k == 3 or k == ord('q'):
            break  # esc to quit

    print(f'FPS = {cam1.get_FPS()}')

    cameras.close()
    # while True:    # def __iter__(self):
    # #     return self
    #     try:
    #         for i, cam in enumerate(cameras.cam_threads):
    #             (grabbed, frame, id) = cam.read()
    #             if grabbed:
    #                 cv2.putText(frame, f"{id}", (100, 205), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 5)
    #                 cv2.imshow(f'{cam.name}', frame)
    #             else:
    #                 pass
    #
    #     except KeyboardInterrupt:
    #         break
    #     except Exception as msg:
    #         logger.error(msg)
    #     else:
    #         time.sleep(0.01)
    #
    #     k = cv2.waitKey(1)
    #     if k == 27 or k == 3 or k == ord('q'):
    #         break  # esc to quit

    # cameras.close()
