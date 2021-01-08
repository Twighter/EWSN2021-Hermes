import numpy as np

class FedSwap():
    def __init__(self, simParameter):
        self.initFedSwap(simParameter)

    def initFedSwap(self, simParameter):
        ueNum = simParameter.ueNum

        self.modelHist = np.ones((ueNum, ueNum))
        for ueIdx in range(ueNum):
            self.modelHist[ueIdx][ueIdx] = 0

    def shuffle(self, modelSetAll, simParameter):
        ueNumAll = simParameter.ueNum
        activeProb = simParameter.activeProb

        ueActive = self.getUeActive(ueNumAll, activeProb)
        ueNum = len(ueActive)
        if(ueNum < 2):
            return np.arange(ueNumAll)

        modelSet = []
        for ueIdx in range(ueNum):
            modelSet.append(modelSetAll[ueActive[ueIdx]])

        modelDist = - self.getModelDist(modelSet, ueNum)

        #minimize:
        #for ueIdx in range(ueNum):
        #    modelDist[ueIdx][ueIdx] = np.inf
        #modelDist = - modelDist

        modelPrefer = self.modify(modelDist, ueActive, ueNum)

        ueTarget = self.myMatching(modelPrefer, ueNum)

        ueTargetAll = np.arange(ueNumAll)
        for ueIdx in range(ueNum):
            ueTargetAll[ueActive[ueIdx]] = ueActive[ueTarget[ueIdx]]

        self.updateModelHist(ueTargetAll, ueNumAll)

        return ueTargetAll

    def getUeActive(self, ueNum, activeProb):
        ueActive = []
        for ueIdx in range(ueNum):
            if(np.random.rand() < activeProb[ueIdx]):
                ueActive.append(ueIdx)

        return ueActive

    def getModelDist(self, modelSet, ueNum):
        modelDist = np.zeros((ueNum, ueNum))
        for ueIdx_1 in range(ueNum):
            for ueIdx_2 in range(ueNum):
                modelDistTemp = 0
                for paramIdx in range(len(modelSet[0]) - 4, len(modelSet[0])):
                    modelParam_1 = modelSet[ueIdx_1][paramIdx]
                    modelParam_2 = modelSet[ueIdx_2][paramIdx]
                    colNum = np.shape(modelParam_1)[1]
                    modelParamMean_1 = np.mean(modelParam_1, axis = 1)
                    modelParamMean_2 = np.mean(modelParam_2, axis = 1)
                    modelParamMeanExt_1 = np.dot(modelParamMean_1[:, np.newaxis], np.ones((1, colNum)))
                    modelParamMeanExt_2 = np.dot(modelParamMean_2[:, np.newaxis], np.ones((1, colNum)))
                    modelParamDelta = ((modelParam_1 - modelParamMeanExt_1) - (modelParam_2 - modelParamMeanExt_2)) ** 2
                    modelDistTemp += np.sum(np.sum(modelParamDelta))
                modelDist[ueIdx_1][ueIdx_2] = modelDistTemp

        return modelDist

    def modify(self, modelDist, ueActive, ueNum):
        modelPrefer = modelDist
        for ueIdx in range(ueNum):
            for modelIdx in range(ueNum):
                modelDist[ueIdx][modelIdx] += self.modelHist[ueActive[ueIdx]][ueActive[modelIdx]] * 10000
        return modelPrefer

    def updateModelHist(self, ueTarget, ueNum):
        for ueIdx in range(ueNum):
            for modelIdx in range(ueNum):
                if ueTarget[modelIdx] == ueIdx:
                    self.modelHist[ueIdx][modelIdx] = 0
                else:
                    self.modelHist[ueIdx][modelIdx] += 1

            modelHistTemp = np.zeros(ueNum)
            for modelIdx in range(ueNum):
                modelHistTemp[modelIdx] = self.modelHist[ueIdx][modelIdx]

            for modelIdx in range(ueNum):
                self.modelHist[ueIdx][ueTarget[modelIdx]] = modelHistTemp[modelIdx]

    def myMatching(self, map, n):

        def createMap(map, threshold):
            mapBin = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    if i != j and map[i][j] > threshold:
                        mapBin[i][j] = 1

            return mapBin

        def dfs(u, map, match, visit):
            for v in range(n):
                if map[u][v] == 1 and visit[v] == 0:
                    visit[v] = 1
                    if(match[v] == -1 or dfs(match[v], map, match, visit)):
                        match[v] = u
                        return True
            return False

        def Hungarian(map):
            match = - np.ones(n).astype(int)
            matchNum = 0

            for i in range(n):
                visit = np.zeros(n)
                if dfs(i, map, match, visit):
                    matchNum += 1

            return matchNum, match

        left = np.min(map)
        right = np.max(map)

        while left + 1e-2 < right:
            mid = (left + right) / 2
            mapBin = createMap(map, mid)
            matchNum, match = Hungarian(mapBin)

            if matchNum == n:
                left = mid
            else:
                right = mid

        matchNum, match = Hungarian(createMap(map, left))
        return match

    def KM(self, map, n):

        def Hungarian(k):
            visit_1[k] = 1
            for i in range(n):
                if(visit_2[i]):
                    continue

                temp = exp_1[k] + exp_2[i] - map[k][i]
                if np.abs(temp) < 1e-5:
                    visit_2[i] = 1
                    if match[i] == -1 or Hungarian(match[i]):
                        match[i] = k
                        return True
                else:
                    if temp < slack[i]:
                        slack[i] = temp

            return False

        match = - np.ones(n).astype(int)
        exp_1 = np.max(map, axis = 1)
        exp_2 = np.zeros(n)

        for i in range(n):
            slack = np.ones(n) * np.inf
            while(True):
                visit_1 = np.zeros(n)
                visit_2 = np.zeros(n)

                if(Hungarian(i)):
                    break

                delta = np.inf
                for j in range(n):
                    if (visit_2[j] == 0) and (slack[j] < delta):
                        delta = slack[j]

                for j in range(n):
                    if visit_1[j] != 0:
                        exp_1[j] -= delta

                    if visit_2[j] != 0:
                        exp_2[j] += delta
                    else:
                        slack[j] -= delta

        return match
