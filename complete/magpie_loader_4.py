
## magpie_loader v4.0 ##
# Version Notes: (December 28th, 2017) This version of the loader normalizaes the data by using scikit-learn's MinMax Scaler found in the preprocessing module.
# This program loads, normalizes, and stadardizes data stored as .csv and exports tuples for use in a neural network.
# This file is a modified version of Michael A. Nielsen's "data_loader" used in "Neural Networks and Deep Learning", Determination Press, 2015.
# The origional file can be found at https://github.com/mnielsen/neural-networks-and-deep-learning.git
# The modified verisions developed by Malcolm Davidson can be found at https://github.com/davidsonnanosolutions/citrine.git
##

#### Libraries
# Standard libraries
import random
import ast

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Global Declerations - store parmeters about the begin and end indexes for usable data within each dataframe
global training_data_start
global training_data_end
global results_data_position
global test_data_start
global test_data_end

training_data_start, training_data_end, results_data_position, test_data_start, test_data_end = [2,-1,-1, 2, 0]

## load_data() function ##
# This function loads traing and test data based on the passed file path.
# The data is normalized and a subset of the training data is set aside as 
# validation data. The function returns three (x,y) tuples composed of numpy
# arrays of data (x) and stability vectors (v). 
##
def load_data(filePath):
    
	
	# File I/O for importing and formatting training data and creating validation data
    with open(filePath, 'rb') as f:

        # Load the training-data.csv file into a (2572,99) pandas dataframe 'raw_df'.
		# Store a sub set of the "raw_df" as "data_df" which omits the stabity vector and the element names.
        raw_df = pd.read_csv(f)
        data_df = raw_df.iloc[:,2:-1]
		
		# Create a temporary datafram "temp_df" to hold normalized version of "data_df".
		# Normalization is accomplished using normalize from  scikit-learn's prepreocessing library.
		# "temp_df" has the same column headers as the origional "data_df"
        temp_df = normalize(data_df)
        temp_df.columns = data_df.columns

		# "norm_df" holds the complete normalized dataframe. This means
		# it has the normalized data stored in "temp_df", the element names origionally
		# in "raw_df", and the stability vector.
        norm_df = raw_df.iloc[:,0:2]
        norm_df = norm_df.join(temp_df)
        norm_df = norm_df.join(raw_df.iloc[:,-1])
		
		# If necessary, the normalized dataframe can be exported to a file for quality control.
        #norm_df.to_csv("/home/spike/citrine/normalized_training_data.csv")

        # Randomly choose 10% of the rows in a non-contiguous fashion to be set aside as validation data.
		# "valIndex" holds n random rows from "norm_df", where n = 0.1*#rows in "norm_df".
		# "validation_df" is the dataframe form of valIndex, sorted and re-indexed.
        valIndex = random.sample(xrange(len(norm_df)),int(round(0.1*len(norm_df))))
        validation_df = pd.DataFrame(norm_df.ix[i,:] for i in valIndex).sort_index(ascending=True)
        validation_df.reset_index()

		# "input_df" is the datafram containg training data to be passed to the network. It is built by 
        # removing validation data from the data to become the "training data" and re-index "norm_df" contiguously.
		# The element labels are also removed producing a [1,96] row vector.
        input_df = norm_df.drop(valIndex)
        input_df.reset_index()
		
		# "input_data" is a (x,y) tuple where x is a float32 np array of training data and y are the associated stability vectors.
        input_data = (input_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32),input_df.iloc[:,results_data_position].values)
		
		# "validation_data" is a (x,y) tuple where x is a float32 np array of validation data and y are the associated stability vectors.
        validation_data = (validation_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32),validation_df.iloc[:,results_data_position].values)
		
	# File I/O for importing and formatting test data.
    with open('/home/spike/citrine/test_data.csv', 'rb') as f:
		
		# Load the test-data.csv file into a (750,99) pandas dataframe 'test_df'.
		# Store a sub set of the "test_df" as "data_df" which omits the stabity vector and the element names.
        test_df = pd.read_csv(f)
        data_df = test_df.iloc[:,2:]

		# See above for descriptions of "test_df" and "norm_df".
        temp_df = normalize(data_df)
        temp_df.columns = data_df.columns

        norm_df = test_df.iloc[:,0:2]
        norm_df = norm_df.join(temp_df)

		# "test_data" is a (x,y) tuple where x is a float32 numpy array of validation data and y are the associated stability vectors.
        test_data = norm_df.iloc[:,test_data_start:].values.astype(np.float32)

    return(input_data, validation_data, test_data)

## load_data_wrapper() function ##	
# This function returns (x,y) tuples of data as list of (96,1) numpy column vectors (x) and a list of stability vectors as
# (11,1) column vectors (y) built from .csv files located at the paased file path.
##	
def load_data_wrapper(filePath):

	
	# Store tuples from load_data() as training data "tr_d", validation data "va_d", and test data "te_d".
    tr_d, va_d, te_d = load_data(filePath)

	# The algorithms used in the network expect a tuple of data composed of a list of column vectors
	# represented with a numpy arrays and a list of stability vectors as column vectors.
    training_inputs = [np.reshape(x, (96, 1)) for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (96, 1)) for x in va_d[0]]
    validation_results = [vectorize(y) for y in va_d[1]]
    validation_data = zip(validation_inputs,validation_results)

    test_data = [np.reshape(x, (96, 1)) for x in te_d]

    return (training_data, validation_data, test_data)

## vectorize() function##
# This returns the string version of the stability vector as a (11,1) column vector.
##
def vectorize(d):

	# Use the Abstract Syntax Trees (ast) to interpret the string version of the stability vector and store it as "d"
	# Define "e" an (11,1) numpy array of 0's to store the stability vector.
    d = ast.literal_eval(d)
    e = np.zeros((11, 1))
	
	# Loop through each element in "d" and assign there value to "e"
    for element in xrange(0,len(d)):
	
        e[element] = d[element]
		
    return(e)

## normalize() function ##
# Applies the sci-kit learn MinMaxScaler to the passed pandas dataframe
# and returns a normalized pandas dataframe.
##
def normalize(raw_df):
    
	
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(raw_df)
    norm_df = pd.DataFrame(scaled_df)

    return(norm_df)
