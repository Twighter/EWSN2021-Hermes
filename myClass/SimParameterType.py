import numpy as np

class SimParameterType:
    def __init__(self):
        self.frameNum = 500
        self.frameSize = 10
        self.slotNum = self.frameNum * self.frameSize
        self.slotTime = 0.001

        self.rbgNum = 6
        self.rbgSize = 16
        self.rbNum = self.rbgNum * self.rbgSize
        self.rbAllocationLimit = 100

        self.subcarrierNum = 12
        self.symNum = 14
        self.dmrsNum = 2

        self.schedulerPeriod = 4
        self.cqiInterval = 200
        self.cqiUpdateCase = [-2, 2]

        self.ueNum = 20

        self.dist = np.zeros(self.ueNum)
        self.distInterval = 0
        for ueIdx in range(self.ueNum):
            self.dist[ueIdx] = 500 - self.distInterval * (ueIdx - self.ueNum / 2)
        self.CQIvsDistance = [[200, 15], [500, 12], [800, 10], [1000, 8], [1200, 7]]

        self.historyMax = 2 *self.ueNum

        #self.appNum = 11
        #self.packetInterval = [7, 2, 12, 9, 4, 4, 8, 2, 7, 4, 4]
        #self.packetSize = [8750, 7500, 7500, 7875, 3000, 7500, 8000, 7500, 8750, 9000, 9000]
        #self.appUe = [0, 0, 0, 0, 1, 1, 2, 2, 2, 3, 3]
        self.appNum = self.ueNum
        self.packetInterval = [1] * self.appNum
        self.packetSize = [100000] * self.appNum
        self.appUe = np.arange(self.ueNum)
        #[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
        #              10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
        #              20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
        #              30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
        #              40, 41, 42, 43, 44, 45, 46, 47, 48, 49]

        self.probSucc = ((self.rbgNum - 1) / self.rbgNum) ** (self.ueNum - 1)
        self.punishRate = - 2 # self.rbgNum/ self.ueNum# -1 / ((1 / self.probSucc) - 1)
        self.rewardNormalizer = 1

        self.activeProb = [1] * self.ueNum

        self.cqiTable = [[0, 0, 0],
                         [2, 78, 0.1523],
                         [2, 193, 0.3770],
                         [2, 449, 0.8770],
                         [4, 378, 1.4766],
                         [4, 490, 1.9141],
                         [4, 616, 2.4063],
                         [6, 466, 2.7305],
                         [6, 567, 3.3223],
                         [6, 666, 3.9023],
                         [6, 772, 4.5234],
                         [6, 873, 5.1152],
                         [8, 711, 5.5547],
                         [8, 797, 6.2266],
                         [8, 885, 6.9141],
                         [8, 948, 7.4063]]

        self.mcsTable = [[2, 120, 0.2344],
                         [2, 193, 0.3770],
                         [2, 308, 0.6016],
                         [2, 449, 0.8770],
                         [2, 602, 1.1758],
                         [4, 378, 1.4766],
                         [4, 434, 1.6953],
                         [4, 490, 1,9141],
                         [4, 553, 2.1602],
                         [4, 616, 2.4063],
                         [4, 658, 2.5703],
                         [6, 466, 2.7305],
                         [6, 517, 3.0293],
                         [6, 567, 3,3223],
                         [6, 616, 3.6094],
                         [6, 666, 3.9023],
                         [6, 719, 4.2129],
                         [6, 772, 4.5234],
                         [6, 822, 4.8164],
                         [6, 873, 5.1152],
                         [8, 682.5, 5.3320],
                         [8, 711, 5.5547],
                         [8, 754, 5.8906],
                         [8, 797, 6.2266],
                         [8, 841, 6.5703],
                         [8, 885, 6.9141],
                         [8, 916.5, 7.1602],
                         [8, 948, 7.0463]]
