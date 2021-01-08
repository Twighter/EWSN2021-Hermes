import numpy as np
import scipy.io as scio
import datetime

from sm_env import SM
from RL_brain_2FC import DeepQNetwork
from FL_brain import FedSwap
from myClass.SimParameterType import SimParameterType

np.random.seed(0)

def run_SM(env, dqn, simParameter):
    slotNum = simParameter.slotNum
    ueNum = simParameter.ueNum

    state = env.reset(simParameter)

    dqnTemp = [[]] * ueNum
    for slotIdx in range(slotNum):
        env.updateEnv(simParameter)

        action = []
        for ueIdx in range(ueNum):
            action.append(dqn[ueIdx].choose_action(state[ueIdx]))

        state_, reward= env.step(state, action, simParameter)

        for ueIdx in range(ueNum):
            dqn[ueIdx].store_transition(state[ueIdx], action[ueIdx], reward[ueIdx], state_[ueIdx])

        state = state_

        if (slotIdx >= 0) and (slotIdx % 10 == 0):
            for ueIdx in range(ueNum):
                dqn[ueIdx].learn()

        if slotIdx % 10 == 0:
            print(slotIdx)
            modelSet = []
            for ueIdx in range(ueNum):
                for ueIdx_2 in range(ueNum):
                    if dqn[ueIdx_2].network_idx == ueIdx:
                        modelSet.append(dqn[ueIdx_2].getParameter())
                        break

        if slotIdx >= 0 and slotIdx % 50 == 0:
            modelSet = []
            for ueIdx in range(ueNum):
                modelSet.append(dqn[ueIdx].getParameter())

            ueTarget = fs.shuffle(modelSet, simParameter)

            for ueIdx in range(ueNum):
                dqnTemp[ueIdx] = dqn[ueIdx]
            for ueIdx in range(ueNum):
                dqn[ueTarget[ueIdx]] = dqnTemp[ueIdx]

            for ueIdx in range(ueNum):
                dqn[ueIdx].clear_transition()

if __name__ == "__main__":
    simParameter = SimParameterType()

    env = SM(simParameter, randSeed = np.random.randint(100))

    dqn = []
    for ueIdx in range(simParameter.ueNum):
       dqnTemp = DeepQNetwork(env.actionNum, env.featureNum, ueIdx,
                              randSeed = np.random.randint(100))
       dqn.append(dqnTemp)

    fs = FedSwap(simParameter)

    startTime = datetime.datetime.now()

    run_SM(env, dqn, simParameter)

    endTime = datetime.datetime.now()
    runtime = (endTime - startTime).seconds + (endTime - startTime).microseconds * 1e-6
    if(runtime < 0):
        runtime += 60
    print('Runtime: %.3f s:'%runtime)

    env.save(simParameter)
    env.plot(simParameter)

