

import os
import numpy as np


class AvgMeter(object):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self,val,n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count



def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def _sigmoid(x):
    return 1/(1+np.exp(-x))



