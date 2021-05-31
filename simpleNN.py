#simple single layer feedforward neural network (perceptron), total of 3 layers,
#use binary digits as inputs, and expect binary digits as output
#use backpropgation via gradient descent to train network to make predictions accurate
#neural networks popular because we have faster computers and more data
#steps: 1. build it, 2. train it, 3. test it

import numpy as np

#function that will map any value to a value between 0 and 1 (a sigmoid)
#this will run in every neuron of our network where data hits it
#creates probabilities out of numbers
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

#input data set (matrix)
#each row is a different training example
#each column represents a different neuron
#we have 4 training examples with 3 input neurons each
X = np.array([[0,0,1],[1,1,1],[1,0,1],[0,1,1]])
#print(X)

#output data set (one output neuron per training example)
y = np.array([[0],[1],[1],[0]])

#Notice the pattern of this data is that the output is the first value of each row.

#seed them to make them deterministic, give numbers the same starting number
#so we get the same sequence of generated numbers every time we run program
np.random.seed(1)

#synapses matrices (connections between layers)
#3 layers, so we need 2 synapses
#each synapses has a random weight assigned to it
syn0 = 2*np.random.random((3,4)) - 1 #create matrix with 3 rows 4 columns with values in the range -1 to 1
#print(syn0)
syn1 = 2*np.random.random((4,1)) - 1

#training step
#iterates over training code to optimize network for given data set
for j in range(60000):

    l0 = X #layer 1 = input data, matrix of 4 rows and 3 columns
    l1 = nonlin(np.dot(l0,syn0)) #prediction step, matrix multiplication between each layer and its synapses
    #print(l1) #prints many matrices, training because synapses is random
    l2 = nonlin(np.dot(l1,syn1)) #prediction of output data, doing the same thing as before

    #error rate from expected output
    l2_error = y - l2
    if(j%10000)==0:
        #printing average error rate at a set interval, want to make sure its going down each time
        print "Error: " + str(np.mean(np.abs(l2_error)))


    #multiply error rate by result of sigmoid function which is used to get the derivative from thr output prediction from layer 2
    #will give a delta which we can use to reduce the error rate of our predictions when we update the synapses every iteration
    l2_delta = l2_error*nonlin(l2, deriv=True)

    #want to see how much error from layer 1 contributes to layer 2
    #this is called #backpropgation
    l1_error = l2_delta.dot(syn1.T) #delta from layer 2 * transpose of synapses 1

    l1_delta = l1_error*nonlin(l1,deriv=True)

    #gradient descent
    #update weights for synapses, to reduce error rate each iteration
    syn1+=l1.T.dot(l2_delta)
    syn0+=l0.T.dot(l1_delta)

print "Output after training"
print l2

#Notice how the output from the network is very close to the expected output of each input by decimals
#Notice how the errors decreased each time
