import cv2
import numpy as np
from imutils import resize

# dt = 0.05
# n=10
# L = 100
# particles=np.zeros(n,dtype=[("position", 'f4' , (2,)),
#                            ("velocity", 'f4' ,(2,)),
#                            ("force", 'f4' ,(2,)),
#                            ("size", 'f4' )])
#
# particles["position"]=np.random.uniform(0,L,(n,2))
# particles["velocity"]=np.zeros((n,2),dtype=np.float32)
# particles["size"]=0.5*np.ones(n,dtype=np.float32)
# particles["force"] = np.random.uniform(-1, 1., (n, 2)) * 1000
#
# XMAX=L
# YMAX=L

class GenCMO:
    def __init__(self, n=10, dt = 0.05, shape=(100,200)):
        self.n = n
        self.dt = dt
        self.shape = shape
        self.L = min(shape)
        # self.XMAX = 2*L
        # self.YMAX = L
        self.cmo = np.zeros(shape, dtype=np.uint8) + 1
        self.random_motion = False

        self.particles = np.zeros(n, dtype=[("pos", 'f4', (2,)), ("vel", 'f4', (2,)),
                                          ("force", 'f4', (2,)), ("mag", 'i1')])

        self.particles["pos"] = np.random.uniform(0, self.L, (n, 2))
        self.particles["vel"] = np.zeros((n, 2), dtype=np.float32)
        self.particles["mag"] = 127 + (np.arange(n, dtype=np.int32)*10) % 100
        self.particles["force"] = np.random.uniform(-1, 1., (n, 2)) * 1000

    def __iter__(self):
        return self

    def __next__(self):
        self.cmo[:] = 0

        if self.random_motion:
            self.particles["force"]=np.random.uniform(-200,200.,(self.n,2))

        self.particles["vel"] = self.particles["vel"] + self.particles["force"]*self.dt
        self.particles["pos"] = self.particles["pos"] + self.particles["vel"]*self.dt
        self.particles["pos"] = self.particles["pos"] % self.shape
        if not self.random_motion:
            self.particles["force"] = self.particles["force"]*0

        cols = self.particles["pos"][:,1].astype(int)
        rows = self.particles["pos"][:,0].astype(int)
        mags = self.particles["mag"].astype(np.uint8)
        # self.cmo[rows,cols] = 200
        for r, c, mag in zip(rows, cols, mags):
            # cv2.rectangle(self.cmo, (c-4, r-4, c+4, r+4), (255, 255, 255), 1)
            cv2.circle(self.cmo, (c, r), 3, int(mag), -1)
        return self.cmo

if __name__ == '__main__':

    gen_cmo = GenCMO(shape=(1080, 1920))
    import sys

    for i in sys.path:
        print(i)

    for i in range(1000):
        # cmo =  update(cmo)
        cmo = next(gen_cmo)
        cv2.imshow('cmo',  resize(cmo, width=500))

        k = cv2.waitKey(100)
        if k == ord('q') or k == 27:
            break

    cv2.waitKey(1000)
    cv2.destroyAllWindows()

