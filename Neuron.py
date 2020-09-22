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
        self.m_gradient = None
        self.eta = 0.15
        self.alpha = 0.5
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

    def calcOutputGradients(self,tagetVal):
        delta = tagetVal - self.m_outputVal
        self.m_gradient = delta * self.transferfunctionDerivative(self.m_outputVal)
    def sumDOW(self,nextLayer):
        sumD = 0.0

        for n in range(0,len(nextLayer) - 1):
            sumD += self.m_outputWeights[n].weight * nextLayer[n].m_gradient
        return sumD
    def updateInputWeight(self,prevLayer):
        for n in range(0,len(prevLayer)):
            neuron = prevLayer[n]
            oldDeltaWeight = neuron.m_outputWeights[self.m_myIndex].deltaWeight

            newDeltaWeight = self.eta + neuron.m_outputVal + self.m_gradient \
                             + self.alpha * oldDeltaWeight

    def calaHiddenGradients(self,nextLayer):
        dow = self.sumDOW(nextLayer)
        self.m_gradient = dow * self.transferfunctionDerivative(self.m_outputVal)
