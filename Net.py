from Neuron import Neuron
class Net:
    def __init__(self,topology):
        self.topology = topology
        self.layers = [[]] #[[]]
        self.layerNums = len(topology)
        for layerNum in range(0,self.layerNums):
            self.layers.append([])
            if layerNum == len(topology) - 1:
                numOutputs = 0
            else:
                numOutputs = topology[layerNum + 1]
            for neuronNum in range(0,topology[layerNum] + 1):
                self.layers[-1].append(Neuron(numOutputs,neuronNum))
                print("Made a Neuron")
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
        pass
    def getReseults(self,resultVals):
        pass