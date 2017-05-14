import numpy as np
import os



FILE = 'dump_'


for i in xrange(10):
    x = np.load(''.join([FILE, str(i)]))
    print x.shape