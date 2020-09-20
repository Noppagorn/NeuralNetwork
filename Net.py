class Net:
    def __init__(self,topology,layers):
        self.topology = topology
        self.layers = layers
        self.layerNums = len(topology)
        for layerNum in range(0,self.layerNums):
            layers.append(None)
            for nuronNum in range(0,len(topology[layerNum])):
                pass
    def feedForward(self,inputVals):
        pass
    def backProp(self,targetVals):
        pass
    def getReseults(self,resultVals):
        pass