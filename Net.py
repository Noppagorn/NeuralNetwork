from Neuron import Neuron
import math
class Net:
    def __init__(self,topology):
        self.topology = topology
        self.layers = [[]] #[[]]
        self.layerNums = len(topology)

        self.m_recentAverageError = None
        self.m_recentAverageSmoothingFactor = None

        for layerNum in range(0,self.layerNums):
            self.layers.append([])
            if layerNum == len(topology) - 1:
                numOutputs = 0
            else:
                numOutputs = topology[layerNum + 1]
            for neuronNum in range(0,topology[layerNum] + 1):
                self.layers[-1].append(Neuron(numOutputs,neuronNum))
                print("Made a Neuron")
        self.layers[-1][-1].m_outputVal = 0
    def feedForward(self,inputVals):
        if len(inputVals) == (len(self.layers[0]) - 1):
            for i in range(0,len(inputVals)):
                self.layers[0][i] = inputVals[i]
            # Forward propragate
            for layerNum in range(1,len(self.layers)):
                prevLayer = self.layers[layerNum - 1]
                for n in range(0,len(self.layers[layerNum]) - 1):
                   self.layers[layerNum][n].feedForward(prevLayer)
    def backProp(self,targetVals):
        outputLayer = self.layers[-1]
        m_error = 0.0

        for n in range(0,len(outputLayer) - 1):
            delta = targetVals[n] - outputLayer[n].m_outputVal
            m_error += delta * delta
        m_error /= (len(outputLayer) - 1)
        m_error = math.sqrt(m_error)

        self.m_recentAverageError = (self.m_recentAverageError * self.m_recentAverageSmoothingFactor + m_error) / (self.m_recentAverageSmoothingFactor + 1.0)
        for n in range(0,len(outputLayer)):
            outputLayer[n].calcOutputGradients[targetVals[n]]
        for layerNum in range(len(self.layers) - 2,0,-1):
            hiddenLayer = self.layers[layerNum]
            nextLayer = self.layers[layerNum + 1]

            for n in range(0,len(hiddenLayer)):
                hiddenLayer[n].calcHiddenGradients(nextLayer)

        for layerNum in range(len(self.layers),0,-1):
            layer = self.layers[layerNum]
            prevLayer = self.layers[layerNum - 1]

            for n in range(0,len(layer)):
                layer[n].updateInputWeights(prevLayer)


    def getReseults(self):
        resultVals = []
        for n in range(0,len(self.layers[-1]) - 1):
            resultVals.append(self.layers[-1][n].m_outputVal)
        return resultVals