import numpy as np

class ContextType:
    def __init__(self, rbgNum):
        self.rbgBitMap = np.zeros(rbgNum)
        self.bitNum = 0
        self.successProb = 0.0
