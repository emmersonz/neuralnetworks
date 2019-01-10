# nn.py
# Emmerson Zhaime
# Artificial Intelligence


import sys
import math
import random

class NN:
    def __init__(self, nodesInLayer):

        # a list: the number of nodes in each layer:
        self._numLayers = len(nodesInLayer)
        self._nodesInLayer = nodesInLayer
        
        # a list: for each layer l except 0, an i x j matrix of weights 
        # from node i in previous layer to node j in this layer.
        # thus:  self._weights[l][i][j] is a weight on a connection in the
        # network.
        # n.b. index l==0 is not used
        self._weights = [None]  # None holds place 0.
        for layer in range(1, self._numLayers):
            self._weights.append(makeMatrix(nodesInLayer[layer-1], 
                                            nodesInLayer[layer]))
        
        # self._in: a 2D list.  self._in[l][j] is the weighted sum of inputs
        # from all nodes i in layer l-1 to node j in layer l.
        #  n.b. index l==0 is not used.
        self._in = [[0] * nodesInLayer[i] for i in range(self._numLayers)]

        # activations:  the outputs from each node in the network..
        # self._activations[l][i] is the activation of node i in layer l.
        self._activations = [[0] * self._nodesInLayer[i] 
                             for i in range(self._numLayers)]

        # self._deltas: a 2D list.  One delta for each node 
        # except those in the first layer.
        # self._deltas[l][i] for l > 0.
        self._deltas = [[0] * nodesInLayer[i] for i in range(self._numLayers)]

    def forward_propagate(self, inputs):
        # give values to input nodes
        assert len(inputs) == self._nodesInLayer[0]
        for i in range(len(inputs)):
            self._activations[0][i] = inputs[i]
        for l in range(1, self._numLayers):
            for j in range(self._nodesInLayer[l]):
                self._in[l][j] = \
                    sum([self._weights[l][i][j] * self._activations[l-1][i] 
                         for i in range(self._nodesInLayer[l-1])])
                self._activations[l][j] = sigma(self._in[l][j])


    def backward_propagate(self, t_values, alpha):
        """ Given a list of desired activations on the last layer
            do a backpropogate and update the weights on the network.
            PRE:  forward_propagate has been done with inputs, and
                  t_values are the desired output for those inputs.  """
        l = self._numLayers-1
        for j in range(self._nodesInLayer[l]):
            err = t_values[j] - self._activations[l][j]
            self._deltas[l][j] = sigma_p(self._in[l][j])*err
        for l in range(l-1, 0, -1):
            for i in range(self._nodesInLayer[l]):
                wsum = sum([self._weights[l+1][i][j] * self._deltas[l+1][j]
                            for j in range(self._nodesInLayer[l+1])])
                self._deltas[l][i] = sigma_p(self._in[l][i]) * wsum

        # updates the weights:
        for l in range(1, self._numLayers):
            for i in range(0, self._nodesInLayer[l-1]):
                for j in range(0, self._nodesInLayer[l]):
                    self._weights[l][i][j] += \
                        alpha * self._activations[l-1][i] * self._deltas[l][j]

    

    def get_error(self, t_values):
        """ return error as sum of squares of differences between desired 
            output (t_values) and corresponding activations on output 
            nodes """
        assert len(t_values) == self._nodesInLayer[self._numLayers-1]
        return sum([(t_values[i] - self._activations[self._numLayers-1][i]) 
                     ** 2 
                    for i in range(len(t_values))])

    def get_output(self):
        """ Return a copy of the activations of the last layer """
        return list(self._activations[self._numLayers-1])

    def print(self):
        print("{}, a NN with {} layers".format(self, self._numLayers))
        for l in range(self._numLayers):
            if l > 0:
                print("Weights from layer {} to layer {}".format(l-1, l))
                for j in range(self._nodesInLayer[l]):
                    print(" To node {} in layer {}".format(j, l))
                    for i in range(self._nodesInLayer[l-1]):
                        print("  " + str(self._weights[l][i][j]) + " ")
                    print()
            print("Activations at layer {}".format(l))
            print(self._activations[l])
            print()
        print("================================")
            
def makeMatrix(rows, cols):
    """ return a random matrix """
    return [[random.random() for _ in range(cols)] for _ in range(rows)]  

def sigma(x):
    # sigmoid function
    return 1 / (1 + math.e ** -x)

def sigma_p(x):
    # derivative of sigmoid function
    return (math.e ** -x) / (1 + math.e ** -x) ** 2


def normalize(v, lower, upper):
    """Returns a normalized value of v"""
    return (v-lower)/(upper - lower)

def makeInputTuple(inputList):
    """ Makes the input tuple """
    if inputList[0] == 1:
        return (inputList[1:], [1,0,0])
    elif inputList[0] == 2:
        return (inputList[1:], [0,1,0])
    elif inputList[0] == 3:
        return (inputList[1:], [0,0,1])

def normalizeIndices(inputlist):
    """Normalize all the data"""
    for i in range(1, len(inputlist[0])):
        mini = inputlist[0][i]
        maxi = inputlist[0][i]
        for j in range(len(inputlist)):
            if inputlist[j][i] < mini:
                mini = inputlist[j][i]
            if inputlist[j][i] > maxi:
                maxi = inputlist[j][i]
        for k in range(len(inputlist)):
            inputlist[k][i] = normalize(inputlist[k][i], mini, maxi)
    return [makeInputTuple(k) for k in inputlist]


print("-------------")

file = open("nnData", "r")
dataList=[]
for line in file:
    myline = line[:-1].split(",")
    intline = [float(i) for i in myline]
    dataList.append(intline)
#print(dataList)
wineData = normalizeIndices(dataList)
wineHalf1 = wineData[:int(len(wineData)/2)]
wineHalf2 = wineData[int(len(wineData)/2):]
wineTraining = wineData[:30] + wineData[60:91] + wineData[130:155]
wineTest = wineData[31:59] + wineData[92:129] + wineData[156:178]

# net = NN([3,3,3])
# net.print()
# net.forward_propagate([0,1,1])
# net.print()


print(dataList[0])
# sys.exit(0)

net = NN([13,6,6,3])

training_set = [([0, 0], [0]),
                ([0, 1], [0]),
                ([1, 0], [0]),
                ([1, 1], [1])]

training_set2 = [([0,0,0], [0,0,1]),
                 ([0,0,1], [0,1,0]),
                 ([0,1,0], [0,1,1]),
                 ([0,1,1], [1,0,0]),
                 ([1,0,0], [1,0,1]),
                 ([1,0,1], [1,1,0]),
                 ([1,1,0], [1,1,1]),
                 ([1,1,1], [0,0,0])]

net.print()
for _ in range(1000):
    err = 0
    for item in wineTraining:
        inputs, outputs = item
        net.forward_propagate(inputs)
        err += net.get_error(outputs)
        net.backward_propagate(outputs, 0.1)
    #print(err)
counter = 1
for item in wineTest:
    inputs, outputs = item
    print(counter, end=": ")
    # print(inputs, end=": ")
    net.forward_propagate(inputs)
    results = net.get_output()
    print(list(round(result, 1) for result in results))
    counter +=1

    #print(net.get_output())

# net2 = NN([3, 3, 3])
# net2.print()
# net2.forward_propagate([0, 1, 1])
# net2.print()
# net2.backward_propagate([1, 0, 0], 0.1)

#print(net2.get_output())
#print(net2.get_error([1, 0, 0]))
