#Program to scrape labels from data set features
import pandas as pd
import sys

#1. Open the file
#2. Save file contents to data frame
#3. remove all elements containing formula B
#3a. Clean up title if necessary
#4. save dataframe to new file

#Holds file path
path = '/home/wizard/data/citirineChallenge'
name = '/training_data.csv'

##LoadFile Function##
#Opens a file in read mode based on the path passed to the method.
#Returns the opened files contents in a variable.
##
def loadFile(filePath,fileName):
	openFile = open(filePath+fileName,'r')
	return openFile

##saveFile Function##
#saves passed data to a file
##
def saveFile(filePath, items):
	openFile = open(filePath+'/labels.csv','w')
	for item in items:
		openFile.write("%s," % item)
	return openFile

##csvToDataFrame Function##
#Imports a CSV file based on the passed path variable.
#Returns the opened files contents in a variable.
##

def csvToDataFrame(filePath,fileName):
	dataFrame = pd.read_csv(filePath+fileName)
	return dataFrame

#1. Loading the Citrine Training data
training_data = csvToDataFrame(path,name)

#2. Save data labels to a list
labels = list(training_data.columns.values)

#3. remove all elements containing formula B
for label in labels:
	if('formulaB' in label):
		labels.remove(label)
	elif('_B' in label):
		labels.remove(label)
	else:
		pass
labels = labels[:-1]
labels = labels[1:]
print(labels)
#4. save dataframe to new file
saveFile(path, labels)



