import numpy as np
np.random.seed(42)
w1 = np.random.randn(32,10)*0.01 # there are 32 neurons and each neuron is connected to all 10 inputs(input layer)
b1= np.zeros((32,1)) #These are the biases for the 32 neurons. They start at zero because there’s no reason to favor one neuron over another before training begins.
w2 = np.random.randn(1,32)*0.01 #This layer takes the 32 outputs from the previous layer and condenses them into 1 final output.
b2 = np.zeros((1,1))
def relu(z):
    return np.maximum(0,z)

def relu_grad(z): #If the input is positive, the slope is 1; if it's zero or negative, the slope is 0.
    return (z > 0).astype(float) 

def forward(X):
    Z1 = w1 @ X + b1 #performs matrix multiplication between each neuron with their weight and adding the bias
    A1 = relu(Z1) # applies relu function so that it becomes non linear
    Z2 = w2 @ A1 + b2 # takes the activated outputs from the hidden layer (A1) and combines them into a single final value (Z2).
    return Z1,A1,Z2 # return all as the model needs to remember the values for updating during backward pass

def backward(X, y, Z1, A1, Z2):
    dZ2 = Z2 - y # differnece between predicted and actual output
    dW2 = dZ2 @ A1.T # the gradient for the output weights. We multiply the error by the values that came out of the hidden layer (A1).
    #.T is used for the transpose of the matrix
    db2 = np.sum(dZ2, axis=1, keepdims=True) #This calculates how much the bias ($b2$) contributed to the error by summing up the differences across all training examples.
    dA1 = w2.T @ dZ2#sending the error "backward" across the weights of Layer 2 to see how much the hidden layer ($A1$) is to blame for the final mistake.
    dZ1 = dA1 * relu_grad(Z1)
    dW1 = dZ1 @ X.T
    db1 = np.sum(dZ1, axis=1, keepdims=True)
    return dW1, db1, dW2, db2
