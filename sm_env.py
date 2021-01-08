import numpy as np
import matplotlib.pyplot as plt
import scipy.io as scio

from myClass.UEType import UEType
from myClass.AppType import AppType

class SM(object):
    def __init__(self, simParameter, randSeed = 0):
        super(SM, self).__init__()
        self.actionNum = simParameter.rbgNum + 1
        self.featureNum = simParameter.rbgNum * 2 + 3

        self.initEnv(simParameter)

        np.random.seed(randSeed)

    def initEnv(self, simParameter):
        ueNum = simParameter.ueNum
        dist = simParameter.dist
        CQIvsDistance = simParameter.CQIvsDistance
        rbNum = simParameter.rbNum
        rbgNum = simParameter.rbgNum
        self.UE = self.initUE(ueNum, dist, CQIvsDistance, rbNum, rbgNum)

        appNum = simParameter.appNum
        packetInterval = simParameter.packetInterval
        packetSize = simParameter.packetSize
        appUe = simParameter.appUe
        self.app = self.initApp(appNum, packetInterval, packetSize, appUe)

        self.cqiIntervalCount = 0
        self.appIntervalCount = np.zeros(appNum)

        self.throughputRecord = []
        self.actionRecord = []
        self.rbgCollisionRecord = []
        self.rbgIdleRecord = []

    def reset(self, simParameter):
        ueNum = simParameter.ueNum
        appNum = simParameter.appNum
        rbgNum = simParameter.rbgNum

        for ueIdx in range(ueNum):
            self.UE[ueIdx].buffer = 0

        self.cqiIntervalCount = 0
        self.appIntervalCount = np.zeros(appNum)

        state = []
        for ueIdx in range(ueNum):
            state.append(self.packState(self.UE[ueIdx], np.zeros(rbgNum), rbgNum, 0, simParameter))
        return state

    def updateEnv(self, simParameter):
        ueNum = simParameter.ueNum
        appNum = simParameter.appNum
        cqiInterval = simParameter.cqiInterval

        self.cqiIntervalCount += 1
        if(self.cqiIntervalCount == cqiInterval):
            self.cqiIntervalCount = 0
            for ueIdx in range(ueNum):
                self.cqiUpdate(self.UE[ueIdx], simParameter.cqiUpdateCase)

        for appIdx in range(appNum):
            appTemp = self.app[appIdx]
            self.appIntervalCount[appIdx] += 1
            if(self.appIntervalCount[appIdx] == appTemp.packetInterval):
                self.appIntervalCount[appIdx] = 0
                self.appUpdate(appTemp)

    def step(self, state, action, simParameter):
        ueNum = simParameter.ueNum
        rbgNum = simParameter.rbgNum
        punishRate = simParameter.punishRate
        rewardNormalizer = simParameter.rewardNormalizer

        state_ = []
        reward = np.zeros(ueNum)
        throughputRecordTemp = []
        for ueIdx in range(ueNum):
            UETemp = self.UE[ueIdx]

            collision = 0
            if(action[ueIdx] != rbgNum):
                for ueIdx_2 in range(ueNum):
                    if(ueIdx != ueIdx_2) and (action[ueIdx] == action[ueIdx_2]):
                        collision = 1
                        break

            bitNum = self.getBitNum(UETemp.cqi, action[ueIdx], simParameter)
            if(collision == 1):
                bitNumSucc = 0
            else:
                if(UETemp.buffer > bitNum):
                    bitNumSucc = bitNum
                else:
                    bitNumSucc = UETemp.buffer

            reward[ueIdx] = bitNumSucc + (bitNum - bitNumSucc) * punishRate
            UETemp.throughput.append(bitNumSucc)
            UETemp.buffer -= bitNumSucc
            throughputRecordTemp.append(bitNumSucc)

            self.UE[ueIdx] = UETemp

            history = state[ueIdx][rbgNum + 2 : 2 * rbgNum + 2]
            state_.append(self.packState(self.UE[ueIdx], history, action[ueIdx], collision, simParameter))

        rbgVisit = np.zeros(rbgNum)
        for ueIdx in range(ueNum):
            if action[ueIdx] < rbgNum:
                rbgVisit[action[ueIdx]] += 1

        idleNum = 0
        collisionNum = 0
        for rbgIdx in range(rbgNum):
            if rbgVisit[rbgIdx] == 0:
                idleNum += 1

            if rbgVisit[rbgIdx] >= 2:
                collisionNum +=1

        self.throughputRecord.append(throughputRecordTemp)
        self.actionRecord.append(action)
        self.rbgCollisionRecord.append(collisionNum)
        self.rbgIdleRecord.append(idleNum)

        return state_, reward / rewardNormalizer

    def save(self, simParameter):
        slotNum = simParameter.slotNum
        slotTime = simParameter.slotTime

        timeAxis = [slotIdx * slotTime for slotIdx in range(0, slotNum)]

        path = './data/Record.mat'
        scio.savemat(path, {'timeAxis' : timeAxis,
                            'throughputRecord' : self.throughputRecord,
                            'actionRecord' : self.actionRecord,
                            'rbgCollisionRecord' : self.rbgCollisionRecord,
                            'rbgIdleRecord' : self.rbgIdleRecord})

    def plot(self, simParameter):
        ueNum = simParameter.ueNum
        slotNum = simParameter.slotNum
        slotTime = simParameter.slotTime

        plt.figure()
        timeAxis = [slotIdx * slotTime for slotIdx in range(0, slotNum)]
        for ueIdx in range(ueNum):
            plt.plot(timeAxis[0: -1: 10], np.array(self.UE[ueIdx].throughput[0: -1: 10]) / slotTime / (1024 * 1024))

        throughputSum = np.zeros(slotNum)
        for ueIdx in range(ueNum):
            throughputSum += np.array(self.UE[ueIdx].throughput)
        plt.plot(timeAxis[0: -1: 10], throughputSum[0: -1: 10] / slotTime / (1024 * 1024))
        plt.show()

        print('Throughput %d Bits'%np.sum(throughputSum))

    def initUE(self, ueNum, dist, CQIvsDistance, rbNum, rbgNum):
        UE = []
        for ueIdx in range(ueNum):
            UETemp = UEType(rbNum, rbgNum)
            UETemp.dist = dist[ueIdx]

            i = 0
            while(i < np.shape(CQIvsDistance)[0] and UETemp.dist > CQIvsDistance[i][0]):
                i += 1
            if(i < np.shape(CQIvsDistance)[0]):
                UETemp.maxCQI = CQIvsDistance[i][1]
            else:
                UETemp.maxCQI = 0

            UETemp.cqi = np.random.randint(1, UETemp.maxCQI+1, rbNum)
            UETemp.history = np.zeros((rbgNum, ueNum * 2))
            UETemp.buffer = 0

            UE.append(UETemp)

        return UE

    def initApp(self, appNum, packetInterval, packetSize, appUe):
        app = []
        for appIdx in range(appNum):
            appTemp = AppType()
            appTemp.packetInterval = packetInterval[appIdx]
            appTemp.packetSize = packetSize[appIdx]
            appTemp.ueIdx = appUe[appIdx]
            app.append(appTemp)

            self.UE[appUe[appIdx]].app.append(appTemp)

        return app

    def packState(self, UE, history, lastAction, collision, simParameter):
        rbgNum = simParameter.rbgNum
        rbgSize = simParameter.rbgSize
        historyMax = simParameter.historyMax

        state = []

        bitNum = []
        for rbgIdx in range(rbgNum):
            bitNum.append(self.getBitNum(UE.cqi, rbgIdx, simParameter) / 1000)
        state.extend(bitNum)

        if(UE.buffer > 0):
            buffer = 1
        else:
            buffer = 0
        state.append(buffer)

        if(lastAction == rbgNum):
            state.append(1)
        else:
            state.append(0)

        for rbgIdx in range(rbgNum):
            if(rbgIdx == lastAction):
                history[rbgIdx] = 0
            else:
                if history[rbgIdx] < historyMax:
                    history[rbgIdx] += 1
        state.extend(history)

        state.append(collision)
        
        return np.array(state)

    def getBitNum(self, cqi, rbgIdx, simParameter):
        cqiTable = simParameter.cqiTable
        mcsTable = simParameter.mcsTable

        rbgSize = simParameter.rbgSize
        rbgNum = simParameter.rbgNum
        subcarrierNum = simParameter.subcarrierNum
        symNum = simParameter.symNum
        dmrsNum = simParameter.dmrsNum

        if(rbgIdx == rbgNum):
            return 0

        rbgStartIdx = rbgIdx * rbgSize
        rbgEndIdx = (rbgIdx + 1) * rbgSize -1
        cqiUsed = cqi[rbgStartIdx : rbgEndIdx]

        cqiMean = int(np.floor(np.mean(cqiUsed)))
        modulation = cqiTable[cqiMean][0]
        coderate = cqiTable[cqiMean][1]

        mcsSelected = 0
        for mcsIdx in range(np.shape(mcsTable)[0]):
            if(modulation == mcsTable[mcsIdx][0])and(mcsTable[mcsIdx][1] <= coderate):
                if(mcsSelected == -1)or(mcsTable[mcsSelected][1] < mcsTable[mcsIdx][1]):
                    mcsSelected = mcsIdx

        bitNum = mcsTable[mcsSelected][2] * rbgSize * subcarrierNum * (symNum - dmrsNum)
        return int(np.floor(bitNum))

    def cqiUpdate(self, UE, cqiUpdateCase):
        for cqiIdx in range(np.shape(UE.cqi)[0]):
            cqiDelta = cqiUpdateCase[np.random.randint(0, len(cqiUpdateCase))]
            UE.cqi[cqiIdx] += cqiDelta
            if(UE.cqi[cqiIdx] > UE.maxCQI):
                UE.cqi[cqiIdx] = UE.maxCQI
            if(UE.cqi[cqiIdx] < 0):
                UE.cqi[cqiIdx] = 0

    def appUpdate(self, app):
        ueIdx = app.ueIdx
        packetSize = app.packetSize * 8
        self.UE[ueIdx].buffer += packetSize
