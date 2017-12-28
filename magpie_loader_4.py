
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

training_data_start, training_data_end, results_data_position, test_data_start, test_data_end = [2,-1,-1, 2, 0]

def load_data():
    
    with open('/home/spike/citrine/training_data.csv', 'rb') as f:

        # load the csv file into a (2572,99) dataframe
        raw_df = pd.read_csv(f)
        data_df = raw_df.iloc[:,2:-1]

        temp_df = normalize(data_df)
        temp_df.columns = data_df.columns

        norm_df = raw_df.iloc[:,0:2]
        norm_df = norm_df.join(temp_df)
        norm_df = norm_df.join(raw_df.iloc[:,-1])

        norm_df.to_csv("/home/spike/citrine/normalized_training_data.csv")

        # randomly choose 10% of the rows to be set aside as validation data
        valIndex = random.sample(xrange(len(norm_df)),int(round(0.1*len(norm_df))))
        validation_df = pd.DataFrame(norm_df.ix[i,:] for i in valIndex).sort_index(ascending=True)
        validation_df.reset_index()

        # remove validation data from the data to become the "training data"
        input_df = norm_df.drop(valIndex)
        input_df.reset_index()

        """
        input_data = []
        input_data.append(input_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32))
        input_data.append(input_df.iloc[:,results_data_position].values)
        """
        input_data = (input_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32),input_df.iloc[:,results_data_position].values)

        """
        validation_data = []
        validation_data.append(validation_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32))
        validation_data.append(validation_df.iloc[:,results_data_position].values)
        """

        validation_data = (validation_df.iloc[:,training_data_start:training_data_end].values.astype(np.float32),validation_df.iloc[:,results_data_position].values)

    with open('/home/spike/citrine/test_data.csv', 'rb') as f:

        test_df = pd.read_csv(f)
        data_df = test_df.iloc[:,2:]

        temp_df = normalize(data_df)
        temp_df.columns = data_df.columns

        norm_df = test_df.iloc[:,0:2]
        norm_df = norm_df.join(temp_df)

        """
        test_data = []
        test_data.append(test_df.iloc[:,test_data_start:test_data_end].values.astype(np.float32))
        """
        test_data = norm_df.iloc[:,test_data_start:].values.astype(np.float32)

    return(input_data, validation_data, test_data)

def load_data_wrapper():

    tr_d, va_d, te_d = load_data()

    """
    print tr_d[0].shape
    print tr_d[0]
    """

    """
    print te_d[0].shape
    print te_d[0]
    """

    training_inputs = [np.reshape(x, (96, 1)) for x in tr_d[0]]
    training_results = [vectorize(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    #print training_data[0:2]

    validation_inputs = [np.reshape(x, (96, 1)) for x in va_d[0]]
    validation_results = [vectorize(y) for y in va_d[1]]
    validation_data = zip(validation_inputs,validation_results)

    test_data = [np.reshape(x, (96, 1)) for x in te_d]

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
