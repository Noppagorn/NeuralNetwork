
class Neuron:
    def __init__(self,numOutputs,myIndex):
        self.numOutputs = numOutputs
        self.myIndex = myIndex
        self.eta = 0.15
        self.alpha = 0.5