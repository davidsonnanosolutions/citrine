## Test Network v1.0 ##
# Version Notes: (December 24th, 2017) 
# Sets up and launches the neural network using the provided training data and network shape
# Developed by Malcolm Davidson. Updates avilable at https://github.com/davidsonnanosolutions/citrine.git
##

#### Librarires
# Third-party libraries
import numpy as np
import pandas as pd

# Custom libraries
# File loader and network
import magpie_loader_4 as mgpl
import network_4 as network

# Global Declerations
global training_data, validation_data, test_data, data_loaded
training_data, validation_data, test_data, = None
data_loaded = False

## menu() function ##
#
##
def menu():
	choice = 0
	print "Welcome to the navigation menu. This menu will allow you to navigate neural network program.\n"
	while choice != 5
		choice = input("Please type the number of your choice:\n 1. Load File\n 2. Load Network\n 3. Train Network\n 4. Test File\n 5. Exit")
		# Load File (1)
		if choice = 1:
			print "Load files Dialogue\n"
			filePath = raw_input("Provide a file path and .csv name to load i.e. C:\data\training_data.csv")
			load_file(filePath)
			
		# Load Network (2)
		elif choice = 2:
		"Load network"
			"Provide a file path and name to load"
			
		# Train Network (3)
		elif choice = 3:
			net_cost
			"Choose a cost function"
			"Default is QuadraticCost"
		"Train network"
			"Provide shape of network [# Input Neurons, hidden layers ..., # Output Neurons]"
			net_shape
			"Default is [96,30,11]"
			"Save results of training?"
				save_trn
				trn_file_name
				trn_file_path
			"Save network after training?"
				save_net
				net_file_path
				"Provide a file path and name to save to"
			method
			"Single training or iterative?"
				# Single Training
					"Provde hyperparameters [Number of epochs, Batch Size, Learning rate]"
					single_hyp
					"Default is [30, 10, 0.2]"
					
				# Iterative Training
					"Which hyperparameter would you like to vary? (Number of epochs (N), Batch Size (B), Learning rate (E))"
					iter_choice
					iter_hyp
					trn_range
					"Provide your training range [Upper Bound, Start, Step Size]"
					
		# Test File (4)
		elif choice = 4:
		"Test file - tests a file loaded as Test Data under the load file option"
			"Provide a path and name to save results"
			"Provide a path and name for a network to use"
		# Exit (5)
			python.quit()

## load_file() function ##
#
##
def load_file(filePath):
	global training_data, validation_data, test_data, data_loaded
	training_data, validation_data, test_data = mgpl.load_data_wrapper(filePath)
	data_loaded = True
return(training_data, validation_data, test_data)

## load_network() function ##
#
##
def load_network():
	# Set up in network file
return()

## train_network() function ##
# Accepts net_shape (list), net_cost (str), method (int)
# net_shape - a list whose first and last element are the number or input and output neurons respectivley.
# Elements between these two are the hidden layers. 
# net_cost - accepts Quadratic or Cross and sets the network to use either the quadratic cost function or
# the cross entropy cost.
# method - accepts 1 or 2, where 1 performs a single training based on fixed hyperparameters or an iterative process
# which tests a range of values for a single hyper parameter. 
##
def train_network(net_shape=[96,30,11], net_cost="Quadratic", method=1, single_hyp=[30,10,0.2], iter_hyp=[30,10,0.2], iter_choice, save_trn, trn_file_name, trn_file_path, trn_range, save_net, net_file_path):


	# Confirm that training data has been loaded.
	if data_loaded = True:
	
		# Set the cost function based on user choice in menu() function
		if net_cost = "Quadratic":
			cost = network.QuadraticCost
		elif net_cost = "Cross":
			cost = network.CrossEntropyCost

		# Creates an instance of the Network class called "net"
		net = network.Network(net_shape, cost)

		# Single Training
		if method = 1:

			# Initilize the weight and biases within the network
			net.large_weight_initializer()

			ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, single_hyp[0], single_hyp[1], single_hyp[2], evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

			epo_df = pd.DataFrame(data=epo)
			epo_df.columns = ['Epoch:']

			cos_df = pd.DataFrame(data=tr_c)
			cos_df.columns = ['Cost:']

			acc_df = pd.DataFrame(data=ev_a)
			acc_df.columns = ['Accuracy:']

			res_df = epo_df.join(cos_df)
			res_df = res_df.join(acc_df)
			
			# Saves the results of training as a .csv with columns: Epoch, Cost, Accuracy at trn_file_name
			if save_trn = 1:
				res_df.to_csv(trn_file_path + "\" + trn_file_name)
			
		# Iterative Training
		elif method = 2:
			
			k = trn_range[1]
			upper = trn_range[0]
			step = trn_range[2]
			
			# Iterate through number of epochs
			if iter_choice = "N":
			
				while k <= upper:
					ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, k, iter_hyp[1], iter_hyp[2], evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
					if save_trn = 1:
						epo_df = pd.DataFrame(data=epo)
						epo_df.columns = ['Epoch:']

						cos_df = pd.DataFrame(data=tr_c)
						cos_df.columns = ['Cost:']

						acc_df = pd.DataFrame(data=ev_a)
						acc_df.columns = ['Accuracy:']

						res_df = epo_df.join(cos_df)
						res_df = res_df.join(acc_df)

						file_name = trn_file_path + "\" + str(k) + ".csv"

						res_df.to_csv(file_name)

					k+=step
			
			# Iterate through batch sizes
			elif iter_choice = "B":
			
				while k <= upper:
					ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, iter_hyp[0], k, iter_hyp[2], evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
					if save_trn = 1:
						epo_df = pd.DataFrame(data=epo)
						epo_df.columns = ['Epoch:']

						cos_df = pd.DataFrame(data=tr_c)
						cos_df.columns = ['Cost:']

						acc_df = pd.DataFrame(data=ev_a)
						acc_df.columns = ['Accuracy:']

						res_df = epo_df.join(cos_df)
						res_df = res_df.join(acc_df)

						file_name = trn_file_path + "\" + str(k) + ".csv"

						res_df.to_csv(file_name)

					k+=step
					
			# Iterate through learning rates
			elif iter_choice = "E":
			
				while k <= upper:
					ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, iter_hyp[0], iter_hyp[1], k, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
					if save_trn = 1:
						epo_df = pd.DataFrame(data=epo)
						epo_df.columns = ['Epoch:']

						cos_df = pd.DataFrame(data=tr_c)
						cos_df.columns = ['Cost:']

						acc_df = pd.DataFrame(data=ev_a)
						acc_df.columns = ['Accuracy:']

						res_df = epo_df.join(cos_df)
						res_df = res_df.join(acc_df)

						file_name = trn_file_path + "\" + str(k) + ".csv"

						res_df.to_csv(file_name)

					k+=step
					
	# If load_file() was skipped, trigger load file dialogue.
	else: load_file()
return()

## test_file() function ##
#
##
def test_File():
return()

# Script of the program begins after this point

print "Welcome to Malcolm Davidson's submission for the Citrine Data Scientist Challenge"
print "\n\nWhere noted, the code used in these files are modified version of Michael A. Nielsen's \"data_loader\" used in \"Neural Networks and Deep Learning\", Determination Press, 2015."
print "The origional file can be found at https://github.com/mnielsen/neural-networks-and-deep-learning.git"
print "Updates to this project can be found at https://github.com/davidsonnanosolutions/citrine.git"
print "\n\nThis program makes use of a neural network who learns by stochastic gradient descent."
print "The program can train networks and evaluate data."

menu()



