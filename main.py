from Net import Net
timeVideo="45:15"
if __name__ == '__main__':
    topology = []
    topology.append(3)
    topology.append(2)
    topology.append(1)
    print(topology)
    myNet = Net(topology)
    inputVals = [1,2,3,4]
    myNet.feedForward(inputVals)

    result = myNet.getReseults()
    print(result)