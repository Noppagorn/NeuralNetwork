from Connection import Connection
import random
import numpy as np
import math
class Neuron:
    def __init__(self,numOutputs,myindex):
        self.m_outputVal = None
        self.m_outputWeights = []
        self.randomWeight = random.random()
        self.m_myIndex = myindex
        for c in range(0,numOutputs):
            self.m_outputWeights.append(Connection())
            self.m_outputWeights[-1].weight = random.random()
    def feefForward(self,prevLayer):
        sumVal = 0

        for n in range(0,len(prevLayer)):
            sumVal += prevLayer[n].m_outputVal * prevLayer[n].m_outputWeights[self.m_myIndex].weight
        self.m_outputVal = self.transferFunction(sumVal)[0]
    def transferFunction(self,sumVal):
        ## มีความเสี่ยงสูงที่จะผิดเพราะ ตัว tanh ของ  C++ ไม่เหมือน Python
        return np.tanh(list(sumVal))
    def transferfunctionDerivative(self,x):
        return 1.0 - math.pow(np.tanh(x)[0],2)