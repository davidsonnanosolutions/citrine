## Neural Network Approach ##
# This program is designed to apply a network 
# composed of neurons using sigmoidal activation functions
# to predicting stable binary compounds. The network learns by applying
# stochastic gradient desent (SGD) after each epoch.
##

import numpy as np
import neuronMath as nm


## Network Class ##
# This class defines a new neural network assuming layer 1 is input and layer 3 is output.
# The number of layers are passed as integers when calling for a new instance of the class. 
# For example net = Network(6,2,1) creates a neural network with 6 input neurones, a hidden
# layer consiting of 2 neurons, and a single output. 
# 
# The network has the following attributes: 
# num_layers - the number of layers in the network (including input and output)
# sizes - the number of neurons per layer
# biases - the 'turn on' threshold for each neuron, randomly initilized
# weights - the weight associated with data exchanged between layers of the network,
# randomly initialized. weights[0] is the weight between layers 1 and 2, while weigts[1]
# is the transition between layers 2 and 3.
##
class Network:


	def __init__(self, sizes):

		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]

	## feedForward method ##
	# Returns the networks output when passed a as input
	# #
	def feedForward(self, a):

		for b, w in zip(*self.biases, self.weights):
			a = nm.sigmoid(np.dot(w, a)+b)
		return a

	## SGD method ##
	# The SGD method trains the neural network by applying stochastic gradient descnt across
	# randomized mini batches. The number of training iterations is set by the number of epochs.
	# training_data is a matrix (x,y) where x is the training data and y are the expected outcomes.
	# When you provide test data, the networks progress is compared to the test data.
	# #
	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):

		if test_data:
			n_test = len(test_data)

		n = len(training_data)

		for j in range(epochs):
			random.shuffle(training_data)
			mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]

			for mini_batch in mini_batches:
					self.update_mini_batch(mini_batch, eta)

			if test_data:
				print("Epoch {0}: {1} / {2}".format(j,self.evaluate(test_data).n_test))
			else:
				print("Epoch {0} comple".format(j))

	## update_mini_batch method ##
	# 
	##
	def update_mini_batch(self, min_batch, eta):

		nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    ## backprop ##
    #
    ##
    def backprop(self, x, y):
    	nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer

        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = nm.sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = self.cost_derivative(activations[-1], y) * nm.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
     
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = nm.sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())

        return (nabla_b, nabla_w)

    ## Evaluate ##
    #
    ##
    def evaluate(self, test_data):
    	test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    ## cost_derivative ##
    #
    ##
    def cost_derivative(self, output_activations, y):
    	return (output_activations-y)



