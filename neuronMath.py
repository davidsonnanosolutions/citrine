## This module defines a sigmoidal function and its derivative ##

import numpy as np

def sigmoid(z):
	return 1.0/(1,0+np.exp(-2))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))