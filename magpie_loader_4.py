
## magpie_loader v4.0
# This version of the loader normalizaes the data by using scikit-learn's MinMax Scaler
##

#### Libraries
# Standard library
import random
import ast

# Third-party libraries
import numpy as np
import pandas as pd
from sklearn import preprocessing

global training_data_start
global training_data_end
global results_data_position
global test_data_start
global test_data_end

training_data_start, training_data_end, results_data_position, test_data_start, test_data_end = [2,-1,-1, 2, 98]

def load_data():
    
    with open('/home/wizard/data/citirineChallenge/training_data.csv', 'rb') as f:

        # load the csv file into a (2572,99) dataframe
        raw_df = pd.read_csv(f)
        #print(raw_df.iloc[0:3,2:-1])
        normalize(raw_df.iloc[:,2:-1])
        #print(raw_df.iloc[1,:])

        
        # randomly choose 10% of the rows to be set aside as validation data
        valIndex = random.sample(xrange(len(raw_df)),int(round(0.1*len(raw_df))))
        validation_df = pd.DataFrame(raw_df.ix[i,:] for i in valIndex).sort_index(ascending=True)
        validation_df.reset_index()

        # remove validation data from the data to become the "training data"
        input_df = raw_df.drop(valIndex)
        input_df.reset_index()

        input_data = []
        input_data.append(input_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32))
        input_data.append(input_df.iloc[:,results_data_position].values)

        validation_data = []
        validation_data.append(validation_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32))
        validation_data.append(validation_df.iloc[:,results_data_position].values)

    with open('/home/wizard/data/citirineChallenge/test_data.csv', 'rb') as f:

        test_df = pd.read_csv(f)

        test_data = []
        test_data.append(test_df.iloc[:,test_data_start:test_data_end].values.astype(np.float64))

    return(input_data, validation_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (96, 1)) for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (96, 1)) for x in va_d[0]]
    validation_results = [vectorize(y) for y in va_d[1]]
    validation_data = zip(validation_inputs,validation_results)
    
    test_data = [np.reshape(x, (96, 1)) for x in te_d[0]]

    print 'Training Inputs: {} '.format(len(training_data))
    return (training_data, validation_data, test_data)

def vectorize(d):
    d = ast.literal_eval(d)
    e = np.zeros((11, 1))
    for element in xrange(0,len(d)):
        e[element] = d[element]
    return(e)

## normalize function
# Try using sk learns minmax scaler
##
def normalize(raw_df):
    
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(raw_df)
    norm_df = pd.DataFrame(scaled_df)

    #print norm_df.iloc[0:3,:]

    return(norm_df)

load_data_wrapper()