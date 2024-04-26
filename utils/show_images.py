__all__ = ['show_img', 'cv2_img_show', 'plot_comparison', 'putText', 'plot_img_and_hist', 'VideoWriter', 'rgb2Gray',
           'image2gray', 'Image', 'vstack']

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import cv2
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

from utils import parameters as pms
import logging
logging.basicConfig(format='%(asctime)-8s,%(msecs)-3d %(levelname)5s [%(filename)10s:%(lineno)3d] %(message)s',
                    datefmt='%H:%M:%S',
                    level=pms.LOGGING_LEVEL)
logger = logging.getLogger(__name__)

matplotlib.rcParams['font.size'] = 16

from imutils import resize


class Image:
    """np.array with metadata."""

    def __init__(self, data, **kwargs):
        self.data = data
        self.metadata = kwargs


def image2gray(im):
    if im.dtype != np.dtype('uint8'):
        print('Changing to uint8')
        im = (im * 255).astype(np.uint8)
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im


# import dask
def show_img(im, txt=None, figsize=None, ax=None, alpha=None, cmap=None, title=None, mode='BGR'):
    """
    Example
        fig, axs = plt.subplots(1, 3, figsize=(12,8))
        for i, ax in enumerate(axs.flatten()):
            show_img(hsv[:,:,i], ax=ax, cmap='magma')

    Args:
        img:
        figsize:
        ax:
        alpha:
        cmap:
        title:
        mode:

    Returns:

    """
    if not ax: fig, ax = plt.subplots(figsize=figsize)
    # if type(img) is np.ndarray   :# or type(img) is dask.array.core.Array:
    #     im = img
    #     md = None
    # else:
    #     im = img.data
    #     md = img.metadata
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., [2, 1, 0]]  # take the first 3 # napari is BGR ??
    if im.ndim == 3 and mode == 'BGR':
        im = im[..., ::-1]  # step -1 channels
    ax.imshow(im, alpha=alpha, cmap=cmap)
    if txt is not None:
        ax.text(15, 25, txt, fontsize=12)
    ax.set_axis_off()
    if title is not None:
        ax.set_title(title)
    # return ax


refPt = []
cropping = False


def cv2_img_show(name, img, width=None, height=None, flags=None, mode='RGB'):
    """ show image in a cv2 namedwindow, you can set the width or height"""
    img = img.astype('uint8')
    try:
        cv2_img_show.count += 1
    except AttributeError:  # will be triggered if this function ahs no property count
        # on first call set all this
        cv2_img_show.count = 0
        cv2.namedWindow(name, flags)
        # setting fullscreen make this show on front
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, flags)
        cv2.imshow(name, img)
        cv2.setMouseCallback(name, _mouse_events)

    if mode=='RGB' and img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = resize(img, width=width, height=height)
    cv2.imshow(name, img)

def vstack(lst:list):
    """ vertically stack images, correct for shape and color """
    try:
        ndim = 0
        maxwidth = 0
        for ls in lst:
            ndim = max(ndim, ls.ndim)
            maxwidth = max(maxwidth, ls.shape[1])
        if ndim == 3:
            for i, ls in enumerate(lst):
                if lst[i].shape[1] != maxwidth:
                    lst[i] = resize(lst[i], width=maxwidth)
                if ls.ndim == 2:
                    lst[i] = cv2.cvtColor(lst[i], cv2.COLOR_GRAY2RGB)

        stack = np.vstack(lst)
    except Exception as e:
        logger.error(f'stacking error {e}')
        stack = np.zeros_like(lst[0])
    return stack

def _mouse_events(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, cropping, curr_image
    # if the left mouse button was clicked, record the starting
    # (x, y) coordinates and indicate that cropping is being
    # performed
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
        cropping = True
    # check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # record the ending (x, y) coordinates and indicate that
        # the cropping operation is finished
        refPt.append((x, y))
        cropping = False
        # draw a rectangle around the region of interest
    elif event == cv2.EVENT_MOUSEMOVE:
        print((x, y))


def plot_comparison(original, filtered, filter_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True, sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')


def plot_img_and_hist(image, figsize=None, axes=None, bins=256, norm=False):
    """Plot an image along with its histogram and cumulative histogram.
    """
    if axes is None:
        if figsize is None: figsize = (10, 5)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    ax_img, ax_hist = axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap='gray')
    ax_img.set_axis_off()

    # Display histogram
    if norm:
        # image_uint8 = (255*(image - np.min(image))/np.ptp(image)).astype(np.uint8)
        image_uint8 = ((image - image.min()) * (1 / (image.max() - image.min()) * 255)).astype('uint8')
        # image_uint8 = ((image /image.max() )* 255).astype('uint8')
    elif image.dtype != np.uint8:
        image_uint8 = img_as_ubyte(image)
    else:
        image_uint8 = image
    ax_hist.hist(image_uint8.ravel(), bins=bins)
    # ax_hist.ticklabel_format(axis='y', style='scientific', scilimits=(0, 0))
    ax_hist.set_xlabel('Pixel intensity')

    xmin, xmax = dtype_range[image_uint8.dtype.type]

    ax_hist.set_xlim(xmin, xmax)
    ax_hist.set_ylabel('Number of pixels')

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image_uint8, bins)
    ax_cdf.plot(bins, img_cdf, 'r')
    return ax_img, ax_hist, ax_cdf


def putText(img, text, row=30, col=10, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 0, 0), thickness=1):
    '''
    Put text on the image
    '''
    if type(img) is np.ndarray:
        cv2.putText(img, text, (col, row), fontFace, fontScale, color, thickness, cv2.LINE_AA)
    else:
        img.metadata = {"text": text, "row": row, "col": col, "fontFace": fontFace, "fontScale": fontScale,
                        "color": color, "thickness": thickness}


def rgb2Gray(img):
    rgb_weights = [0.2989, 0.5870, 0.1140]  # Rec. 601 Color Transform
    return np.dot(img[..., :3], rgb_weights)


class VideoWriter:
    def __init__(self, filename='_autoplay.mp4', fps=30.0, **kw):
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)
        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)
        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()
        if self.params['filename'] == '_autoplay.mp4':
            self.show()

    def show(self, **kw):
        self.close()
        fn = self.params['filename']
        display(mvp.ipython_display(fn, **kw))

if __name__ == '__main__':
    import cv2
    from cv2 import cv2
    from pathlib import Path

    home = str(Path.home())
    path = home + "/temp/"
    video = VideoWriter(path + 'out.mp4', 5.0)
    for i in range(100):
        img = (np.random.rand(100,100,3) * 255).astype('uint8')
        video.add(img)

    video.close()