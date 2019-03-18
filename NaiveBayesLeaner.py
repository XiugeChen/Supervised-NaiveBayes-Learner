# Project1 for course Machine Learning COMP30027 at Unimelb
# Coordinator: Tim Baldwin
# Supervised Naive Bayes learner
# Xiuge Chen
# 2019.03.15

import os

##### SYSTEM SETTINGS #####

# data folder path
FOLDER_PATH = "./2019S1-proj1-data/"
# file name that contains header information
HEADER_FILE = "headers.txt"

##### END SYSTEM SETTINGS #####

#### MAIN FUNCTION ####
# main function that initialize and execute this program
def main(): 
    # get all file paths and header mapping for each file
    fileNames = getFileNames(FOLDER_PATH)
    
    for fileName in fileNames:
        dataSet = preprocess(fileName)
        
        model = train(dataSet)
        
        print(fileName, model)
    
    return
    
#### END MAIN FUNCTION ####
    
#### DATA PROCESSING ####
# opens a data file, and converts it into a usable format dataset
# INPUT: fileName that contains data
# OUTPUT: dataset, a list of instance, where instance is a 2-tuple 
# (a list of attribute values, and a class label attribute)
def preprocess(fileName):
    dataSet = []
    
    # get headers, transform into instance and append it to the first line of dataset
    headerArray = getHeader(fileName)
    dataSet.append((headerArray[:len(headerArray) - 1], headerArray[len(headerArray) - 1]))
    
    # get data, transform into instance and append it to dataset
    dataFile = open(FOLDER_PATH + fileName, 'r')
    for line in dataFile:
        line = line.rstrip('\n')
        dataArray = line.split(',')
        
        dataSet.append((dataArray[:len(dataArray) - 1], dataArray[len(dataArray) - 1]))
    
    return dataSet

# given a fileName, return the header that decripts to this data file
# INPUT: file name
# OUTPUT: array of elements in header
def getHeader(fileName):
    headerFile = open(FOLDER_PATH + HEADER_FILE, 'r') 
    lines = headerFile.readlines()
    
    index = lines.index(fileName + '\n')   
    header = lines[index + 2]
    header = header.rstrip('\n')
    headerArray = header.split(',')
    
    return headerArray

# given a folder that contains
def getFileNames(folderPath):
    filePaths = []
    
    for dicName in os.listdir(folderPath):
        if dicName.startswith('.') or not dicName.endswith(".csv"):
            continue
        
        filePaths.append(dicName)
    
    return filePaths
    
#### END DATA PROCESSING ####

#### TRAINING STAGE ####
# calculate probabilities (prior, posteriors) from the training data, to build a Naive Bayes (NB) model
# missing value handling: not contribute to the counts/probability estimates
# Also add Laplace smoothing
# INPUT: usable format dataSet with header as first row
# OUTPUT: model, a 2-tuple contains a dictionary of normalised counts (probabilities) representing the class 
# distribution P(c), and a dictionary (one per class) of dictionaries (one per attribute) of dictionaries (keys are attribute 
# values, values are counts or probabilities) representing the conditional probabilities P(a|c);
def train(dataSet):
    # class distribution and conditional probabilities
    Pc, Pac = {}, {}  
    classCount, dataHeader = 0, dataSet[0][0]

    for instance in dataSet[1:]:
        # count priors and initialize posteriors
        classTag, attributes = instance[1], instance[0]
        
        if classTag in Pc:
            Pc[classTag] = Pc[classTag] + 1
            classCount += 1
                
        else:
            Pc[classTag], Pac[classTag] = 1, {}
        
        # count posteriors, Pac represent conditional probabilities
        for i in range(0, len(attributes)):
            attributeName, attributeValue = dataHeader[i], attributes[i]
        
            if attributeName in Pac[classTag]:
                if attributeValue in Pac[classTag][attributeName]:
                    Pac[classTag][attributeName][attributeValue] = Pac[classTag][attributeName][attributeValue] + 1
                else:
                    Pac[classTag][attributeName][attributeValue] = 1
            else:
                Pac[classTag][attributeName] = {}
                Pac[classTag][attributeName][attributeValue] = 1
                
    # normalise counts to get posteriors probabilities of each values of each attributes of each classes
    # eliminate missing value and add Laplace smoothing
    for singleClass in Pac.keys():
        for attribute in Pac[singleClass].keys():
            valueKeys = Pac[singleClass][attribute].keys()
            numInstance = Pc[singleClass] + len(valueKeys) + 1
            
            # if there is missing value, not contribute to the counts/probability estimates
            if '?' in valueKeys:
                numInstance = numInstance - Pac[singleClass][attribute]['?'] - 1
                del Pac[singleClass][attribute]['?']
            
            for valueKey in valueKeys:
                # Laplace smoothing
                Pac[singleClass][attribute][valueKey] = (Pac[singleClass][attribute][valueKey] + 1) / numInstance
            
            # add posteriors probability of unknown/new values
            Pac[singleClass][attribute]["unknow"] = 1 / numInstance
    
    # normalise counts to get priors probabilities of each classes
    for singleClass in Pc.keys():
        Pc[singleClass] = Pc[singleClass] / classCount
    
    return (Pc, Pac)
    
#### END TRAINING STAGE ####

#### PREDICTING STAGE ####

# predict classes distribution for the test dataset
# INPUT: the Naive Bayes model, usable format dataSet with header at first row
# OUTPUT: a list of class labels
def predict(model, dataset):
    dataHeader = dataSet[0][0]

    for instance in dataset[1:]:
        continue
    
    return
    
#### END PREDICTING STAGE ####

# OUTPUT: void, but prints the necessary evaluation metric information
def evaluate():
    return

# OUTPUT: a list of values of Information Gain (one per attribute)
def info_gain():
    return
    
# make the main function work
if __name__ == "__main__":
    main()