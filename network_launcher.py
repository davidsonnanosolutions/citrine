## Test Network v1.0 ##
# Version Notes: (December 24th, 2017) 
# Sets up and launches the neural network using the provided training data and network shape
# Developed by Malcolm Davidson. Updates avilable at https://github.com/davidsonnanosolutions/citrine.git
##

#### Librarires
# Standard libraries
import ast

# Third-party libraries
import numpy as np
import pandas as pd
from pyfiglet import figlet_format

# Custom libraries
# File loader and network
import magpie_loader_4 as mgpl
import network2_4 as network

# Global Declerations
global training_data, validation_data, test_data, test_labels, data_loaded, loaded_network, network_loaded
training_data, validation_data, test_data, test_labels, loaded_network = None, None, None, None, None
data_loaded = False
network_loaded = False

## menu() function ##
#
##
def menu():

	net_shape = None
	net_cost = None
	save_trn = None
	trn_file_path = None
	trn_file_name = None
	save_net = None
	net_file_path = None
	method = None
	single_hyp = None
	iter_choice = None
	iter_range = None
	iter_hyp = None

	choice = 0
	print "Welcome to the navigation menu. This menu will allow you to navigate neural network program.\n"

	while choice != 5:
		choice = int(raw_input("Please type the number of your choice:\n 1. Load File\n 2. Load Network\n "
							   "3. Train Network\n 4. Test File\n 5. Exit"))

		# Load File (1)
		if choice == 1:
			print "Load Files Dialogue\n"
			filePath = raw_input("Provide a file path to load training i.e. /home/user/data\n "
								 "Assumes name is training_data.csv\n")
			if filePath == "":
				filePath = "/home/wizard/citrine"
			load_file(filePath)
			
		# Load Network (2)
		elif choice == 2:
			print "Load Network Dialogue\nLoaded networks are currently only used to evalute test data and cannot be trained further\n\n"
			net_file_path = raw_input("Provide a file path and name to load i.e. /home/user/data/_networks/my_network.nwk")
			if net_file_path == "":
				net_file_path = "/home/wizard/data/network.nwk"

			load_network(net_file_path)

		# Train Network (3)
		elif choice == 3:
			print "Network Training Dialogue\n"

			net_cost = raw_input("Choose a cost function:\n (quad)ratic\n (cross) entropy\n Please type the text in parentheses:")

			if net_cost == "":
				net_cost = "quad"

			net_shape = raw_input("Provide shape of network [# Input Neurons, hidden layers ..., # Output Neurons]\nDefault is [96,30,11]")

			if net_shape == "":
				net_shape = [96,30,11]
			else:
				net_shape = ast.literal_eval(net_shape)
		
			save_trn = raw_input("Save results of training (accuracy and cost)?\n 1. Yes\n 2. No")

			if save_trn == "":
				save_trn = "2"

			if int(save_trn) == 1:
				trn_file_path = raw_input("Provide a file path to save to i.e. /home/user/data/_results")
				trn_file_name = raw_input("Provide a file name to save to i.e. epochs_96_30_11_k30_m10_e0.5.csv")
			elif int(save_trn) == 2:
				trn_file_path = None
				trn_file_name = None
				pass

			save_net = raw_input("Save trained network after training?\n 1. Yes\n 2. No")

			if save_net == "":
				save_net = "2"

			if int(save_net) == 1:
				net_file_path = raw_input("Provide a file path and name to save to i.e. /home/user/data/_networks/my_network\nFile will be saved as .nwk")
			elif int(save_net) == 2:
				net_file_path = None
				pass

			method = raw_input("Please type the number of your choice:\n 1. Train with fixed hyperparameters\n 2. Train while iterating over a hyperparameter")

			if method == "":
				method = 1

			# Single Training
			if int(method) == 1:
				single_hyp = raw_input("Provide hyperparameters [Number of epochs, Batch Size, Learning rate]\nDefault is [30, 10, 0.2]")

				if single_hyp == "":
					single_hyp = [30,10,0.2]
				else:
					single_hyp = ast.literal_eval(single_hyp)

				train_network(net_shape, net_cost, save_trn, trn_file_path, trn_file_name, save_net, net_file_path, method, single_hyp)

			# Iterative Training
			elif int(method) == 2:

				iter_choice = raw_input("Which hyperparameter would you like to vary? (Number of epochs (N), Batch Size (B), Learning rate (E))")

				iter_range = ast.literal_eval(raw_input("Provide your training range [Upper Bound, Start, Step Size]\ni.e. training eta from 0.1 to 1 in 0.1 steps"))

				iter_hyp = raw_input("Provide hyperparameters including that to be trained [Number of epochs, Batch Size, Learning rate]\nDefault is [30, 10, 0.2]")

				if iter_hyp == "":
					iter_hyp = [30,10,0.2]
				else:
					iter_hyp = ast.literal_eval(iter_hyp)

				train_network(net_shape, net_cost, save_trn, trn_file_path, trn_file_name, save_net, net_file_path, method, single_hyp, iter_choice, iter_range, iter_hyp)

		# Test File (4) - work in progress
		elif choice == 4:
			if network_loaded == True:
				print "Test File Dialogue\n"

				filePath = raw_input("Provide a file path to load test data i.e. /home/user/data\n "
										 "Assumes name is test_data.csv\n")

				if filePath == "":
					filePath = "/home/wizard/citrine"

				resultsFilePath = raw_input("Provide a file path and name to save results i.e. /home/user/data/me_results.csv\n ")

				if resultsFilePath == "":
					resultsFilePath = "/home/wizard/data/test_results.csv"

				test_file(filePath,resultsFilePath)

			else:
				print "Please load a network first"

		# Exit (5)

## load_file() function ##
#
##
def load_file(filePath):

	global training_data, validation_data, data_loaded
	training_data, validation_data = mgpl.load_data_wrapper(filePath)
	data_loaded = True
	return(training_data, validation_data)

## load_file() function ##
#
##
def load_test_file(filePath):

	global test_data, test_labels
	test_data, test_labels = mgpl.load_test_data_wrapper(filePath)
	return(test_data, test_labels)

## load_network() function ##
#
##
def load_network(filePath):

	global loaded_network, network_loaded

	loaded_network = network.load(filePath)
	network_loaded = True
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
def train_network(net_shape=[96,30,11], net_cost="quadratic", save_trn=None, trn_file_path=None, trn_file_name=None, save_net=None, net_file_path=None, method=1, single_hyp=[30,10,0.2], iter_choice=None, iter_range=None, iter_hyp=[30,10,0.2]):

	# Confirm that training data has been loaded.
	if data_loaded == True:
	
		# Set the cost function based on user choice in menu() function
		if net_cost == "quad":
			cost = network.QuadraticCost
		elif net_cost == "cross":
			cost = network.CrossEntropyCost

		# Creates an instance of the Network class called "net"
		net = network.Network(net_shape, cost)

		# Single Training
		if int(method) == 1:

			# Initilize the weight and biases within the network
			net.large_weight_initializer()

			ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, single_hyp[0], single_hyp[1], single_hyp[2], evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True,plot_results=True)

			epo_df = pd.DataFrame(data=epo)
			epo_df.columns = ['Epoch:']

			cos_df = pd.DataFrame(data=tr_c)
			cos_df.columns = ['Cost:']

			acc_df = pd.DataFrame(data=ev_a)
			acc_df.columns = ['Accuracy:']

			res_df = epo_df.join(cos_df)
			res_df = res_df.join(acc_df)

			# Saves the results of training as a .csv with columns: Epoch, Cost, Accuracy at trn_file_name
			if int(save_trn) == 1:
				res_df.to_csv(trn_file_path + '/' + trn_file_name)

			if int(save_net) == 1:
				net.save(net_file_path + ".nwk")
			
		# Iterative Training
		elif int(method) == 2:
			
			k = iter_range[1]
			upper = iter_range[0]
			step = iter_range[2]
			
			# Iterate through number of epochs
			if iter_choice == "N":
			
				while k <= upper:
					ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, k, iter_hyp[1], iter_hyp[2], evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)

					if int(save_trn) == 1:
						epo_df = pd.DataFrame(data=epo)
						epo_df.columns = ['Epoch:']

						cos_df = pd.DataFrame(data=tr_c)
						cos_df.columns = ['Cost:']

						acc_df = pd.DataFrame(data=ev_a)
						acc_df.columns = ['Accuracy:']

						res_df = epo_df.join(cos_df)
						res_df = res_df.join(acc_df)

						file_name = trn_file_path + "/" + str(k) + ".csv"

						res_df.to_csv(file_name)

					if int(save_net) == 1:
						net.save(net_file_path + str(k) + ".nwk")

					k+=step
			
			# Iterate through batch sizes
			elif iter_choice == "B":
			
				while k <= upper:
					ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, iter_hyp[0], k, iter_hyp[2], evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
					if int(save_trn) == 1:
						epo_df = pd.DataFrame(data=epo)
						epo_df.columns = ['Epoch:']

						cos_df = pd.DataFrame(data=tr_c)
						cos_df.columns = ['Cost:']

						acc_df = pd.DataFrame(data=ev_a)
						acc_df.columns = ['Accuracy:']

						res_df = epo_df.join(cos_df)
						res_df = res_df.join(acc_df)

						file_name = trn_file_path + "/" + str(k) + ".csv"

						res_df.to_csv(file_name)

					if int(save_net) == 1:
						net.save(net_file_path + str(k) + ".nwk")

					k+=step
					
			# Iterate through learning rates
			elif iter_choice == "E":
			
				while k <= upper:
					ev_c, ev_a, tr_c, tr_a, epo = net.SGD(training_data, iter_hyp[0], iter_hyp[1], k, evaluation_data=validation_data, monitor_evaluation_accuracy=True, monitor_training_cost=True)
					if int(save_trn) == 1:
						epo_df = pd.DataFrame(data=epo)
						epo_df.columns = ['Epoch:']

						cos_df = pd.DataFrame(data=tr_c)
						cos_df.columns = ['Cost:']

						acc_df = pd.DataFrame(data=ev_a)
						acc_df.columns = ['Accuracy:']

						res_df = epo_df.join(cos_df)
						res_df = res_df.join(acc_df)

						file_name = trn_file_path + "/" + str(k) + ".csv"

						res_df.to_csv(file_name)

					if int(save_net) == 1:
						net.save(net_file_path + str(k) + ".nwk")

					k+=step

	# If load_file() was skipped, trigger load file dialogue.
	elif data_loaded == False: print "Please load training data first"
	return()

## test_file() function ##
#
##
def test_file(filePath, resultsFilePath):

	# Sets the global test data to data stored within the path provided
	load_test_file(filePath)
	results = []

	for i in xrange(0,len(test_data)):
		#results.append(network.approximate(loaded_network.feedforward(test_data[i]),55).tolist())
		l = network.approximate(loaded_network.feedforward(test_data[i]),55).tolist()

		results_str = ",".join(str(x) for x in l)
		results_str = results_str.translate(None,"[")
		results_str = results_str.translate(None, "]")
		results_str = "[" + results_str + "]"

		results.append(results_str)

	test_results = pd.DataFrame(data=results,columns=["Stability Vector"])

	results_df = test_labels.join(test_results)

	results_df.to_csv(resultsFilePath)

	return()

# Script of the program begins after this point
print(figlet_format('Citrine Challenge', font='starwars'))
print "Welcome to Malcolm Davidson's submission for the Citrine Data Scientist Challenge"
print "\n\nWhere noted, the code used in these files are modified version of Michael A. Nielsen's \"data_loader\" used in \"Neural Networks and Deep Learning\", Determination Press, 2015."
print "The origional file can be found at https://github.com/mnielsen/neural-networks-and-deep-learning.git"
print "Updates to this project can be found at https://github.com/davidsonnanosolutions/citrine.git"
print "\n\nThis program makes use of a neural network who learns by stochastic gradient descent."
print "The program can train networks and evaluate data.\n\n"

menu()



