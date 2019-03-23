# Project1 for course Machine Learning COMP30027 at Unimelb
# Coordinator: Tim Baldwin
# Supervised Naive Bayes learner
# Xiuge Chen
# 2019.03.15

import os
import sys
import math

##### SYSTEM SETTINGS #####

# data folder path
FOLDER_PATH = "./2019S1-proj1-data/"
# file name that contains header information
HEADER_FILE = "headers.txt"
# epsilon smoothing value
EPSILON = sys.float_info.epsilon

##### END SYSTEM SETTINGS #####
    
#### DATA PROCESSING ####
# opens a data file, and converts it into a usable format dataset with proper header at its first row
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

# partition given data set into training and testing data
# INPUT: dataset
# OUTPUT: list of training dataset, list of testing dataset
def partition_testOnTrain(dataSet):   
    return dataSet.copy(), dataSet.copy()
    
#### END DATA PROCESSING ####

#### TRAINING STAGE ####
# calculate probabilities (prior, posteriors) from the training data, to build a Naive Bayes (NB) model
# missing value handling: not contribute to the counts/probability estimates
# INPUT: usable format dataSet with proper header as first row
# OUTPUT: model, a 2-tuple contains a dictionary of normalised counts (probabilities) representing the class 
# distribution P(c), and a dictionary (one per class) of dictionaries (one per attribute) of dictionaries (keys are attribute 
# values, values are counts or probabilities) representing the conditional probabilities P(a|c);
def train(dataSet):
    # class distribution and conditional probabilities
    pc, pac = {}, {}  
    dataHeader = dataSet[0][0]

    for instance in dataSet[1:]:
        # count priors and initialize posteriors
        classTag, attributes = instance[1], instance[0]
        
        if classTag in pc:
            pc[classTag] += 1
        else:
            pc[classTag], pac[classTag] = 1, {}
        
        # count posteriors, pac represent conditional probabilities
        for i in range(0, len(attributes)):
            attributeName, attributeValue = dataHeader[i], attributes[i]
        
            if attributeName in pac[classTag]:
                if attributeValue in pac[classTag][attributeName]:
                    pac[classTag][attributeName][attributeValue] += 1
                else:
                    pac[classTag][attributeName][attributeValue] = 1
            else:
                pac[classTag][attributeName] = {}
                pac[classTag][attributeName][attributeValue] = 1
                
    # normalise counts to get posteriors probabilities of each values of each attributes of each classes
    # eliminate missing value 
    for singleClass in pac.keys():
        for attribute in pac[singleClass].keys():
            valueKeys = pac[singleClass][attribute].keys()
            numInstance = pc[singleClass]
            
            # if there is missing value, not contribute to the counts/probability estimates
            if '?' in valueKeys:
                numInstance = numInstance - pac[singleClass][attribute]['?']
                del pac[singleClass][attribute]['?']
            
            for valueKey in valueKeys:
                pac[singleClass][attribute][valueKey] = pac[singleClass][attribute][valueKey] / numInstance
    
    # normalise counts to get priors probabilities of each classes
    for singleClass in pc.keys():
        pc[singleClass] = pc[singleClass] / len(dataSet[1:])
    
    return (pc, pac)
    
#### END TRAINING STAGE ####

#### PREDICTING STAGE ####

# predict classes distribution for the test dataset, for each probabilities use log-transformation to prevent underflow, for new values using epsilon smoothing method
# INPUT: the Naive Bayes model, usable format dataSet with proper header at first row
# OUTPUT: a list of class labels
def predict(model, dataSet):
    dataHeader, predictClass = dataSet[0][0], []
    pc, pac = model

    for instance in dataSet[1:]:
        maxClass, maxProb = "", float('-inf')
        attributes = instance[0]
        
        for singleClass in pac.keys():
            # calculate probabilities for each class, replace 0 with epsilon
            prob = math.log(pc[singleClass])
            
            for i in range(0, len(attributes)):
                attributeName, attributeValue = dataHeader[i], attributes[i]
                
                if attributeValue in pac[singleClass][attributeName].keys():
                    prob += math.log(pac[singleClass][attributeName][attributeValue])
                else:
                    prob += math.log(EPSILON)
            
            if prob > maxProb:
                maxClass, maxProb = singleClass, prob

        predictClass.append(maxClass)
        
    return predictClass
    
#### END PREDICTING STAGE ####

#### EVALUATION STAGE ####

# output evaluation metric(s) for single model and test data, or sufficient information so that they can be easily calculated by hand
# INPUT: trained model, usable dataset with header for testing
# OUTPUT: void, but prints the necessary evaluation metric information
# print example:
#   A B
# A 2 1
# B 0 3
# meaning: 2 instances of class A are correctly identified as A, 1 is mistakenly identified as B, all 3 instances of B are correctly identified as B, none of them is mistakenly identified as A
def evaluate(model, testSet):
    # get correct label array
    correctLabels = []
    for instance in testSet[1:]:
        correctLabels.append(instance[1])
    
    resultLabels = predict(model, testSet)
    
    # get result 2d map, primary key is each primary class, secondary key and value are the number of other class primary class has been identified into
    # ex. {'A': {'A': 2, 'B': 1}} means for total 3 test instances of A, 2 of them are correctly identified as A itself, 1 is mistakenly identified as B
    metricMap = {}
    for i in range(0, len(correctLabels)):
        correctLabel, resultLabel = correctLabels[i], resultLabels[i]
        
        if correctLabel in metricMap.keys():
            if resultLabel in metricMap[correctLabel].keys():
                metricMap[correctLabel][resultLabel] += 1
            else:
                metricMap[correctLabel][resultLabel] = 1
        else: 
            metricMap[correctLabel] = {}
            metricMap[correctLabel][resultLabel] = 1
            
    # transform the result map to output matrix format and output results       
    print("Class Idendification Matrix:")
    print("\t" + "".join(key + "\t" for key in metricMap.keys()))
    
    for priKey in metricMap.keys():
        countList = [priKey]
        
        for secKey in metricMap.keys():
            if secKey in metricMap[priKey].keys():
                countList.append(metricMap[priKey][secKey])
            else:
                countList.append(0)
                
        print("".join(str(count) + "\t" for count in countList))
        
    return

# calculate the Information Gain (IG) for one (or each) attribute, relative to the class distribution
# INPUT: A trained model
# OUTPUT: a dictionary maps each attribute to its values of Information Gain
def info_gain(model):
    pc, pac = model
    ig, attributes_values, output = 0, {}, {}
    
    # get entire attributes and value mapping list
    attributes = []
    for eachclass in pac:
        attributes.extend(pac[eachclass].keys())  
    # get unique attributes set
    attributes = list(set(attributes))
    
    for attribute in attributes:
        values = []
        
        for eachclass in pac:
            values.extend(pac[eachclass][attribute].keys())
        
        values = list(set(values))
        attributes_values[attribute] = values

    # entropy for root, the entropy before splitting the tree using the attribute’s values
    for eachclass in pc.keys():
        ig -= pc[eachclass] * math.log(pc[eachclass], 2)
    
    for attribute in attributes_values.keys():
        meanInfo = 0
        # the weighted average of the entropy over the children after the split (Mean Information)
        # Mean Information (attribute a) = sum_v( P(value v) * H(value v) )
        for value in attributes_values[attribute]:
            # P(a=v), prob_av
            prob_av = 0
            for eachclass in pac.keys():
                if attribute in pac[eachclass].keys() and value in pac[eachclass][attribute].keys():
                    prob_av += pac[eachclass][attribute][value] * pc[eachclass]
        
            # H(a=v) =  - sum_c( P(class c | a=v) * log(P(c | a=v)) ), h_av
            # P(c | a=v) = (P(a=v | c) * P(c)) / P(a=v), prob_c_av
            h_av = 0
            for eachclass in pac.keys():
                if attribute in pac[eachclass].keys() and value in pac[eachclass][attribute].keys():
                    prob_c_av = (pac[eachclass][attribute][value] * pc[eachclass]) / prob_av
                    h_av -= prob_c_av * math.log(prob_c_av, 2)          
                
            meanInfo += prob_av * h_av
            
        output[attribute] = ig - meanInfo

    return output
    
#### END EVALUATION STAGE ####

#### MAIN FUNCTION ####
# main function that initialize and execute this program
def main(): 
    # get all file paths and header mapping for each file
    fileNames = getFileNames(FOLDER_PATH)
    
    for fileName in fileNames:
        print("File: " + fileName)
    
        dataSet = preprocess(fileName)
        
        trainSet, testSet = partition_testOnTrain(dataSet)
        
        model = train(trainSet)
        
        result = predict(model, testSet)
        
        evaluate(model, testSet)
        
        iglist = info_gain(model)
        
        print(iglist)
    
    return
    
# make the main function work
if __name__ == "__main__":
    main()
    
#### END MAIN FUNCTION ####