import numpy as np
import copy
def readData():
    #filename = input('Type in the name of the file to test: ')
    filename = 'CS170_Small_Data__96.txt'
    data = np.loadtxt(filename)
    #Get amount of rows and columns from our np data-structure
    #https://www.geeksforgeeks.org/find-the-number-of-rows-and-columns-of-a-given-matrix-using-numpy/
    return np.shape(data)[0], np.shape(data)[1], data

def accuracy(data, features, feature_to_add): #k cross fold algorithm in lecture
    number_correctly_classified = 0
    rows = np.shape(data)[0]
    for i in range(0, rows):
        object_to_classify = 0
        label_object_to_classify = data[i][0]
        print('Looping over i at the ' + str(i + 1) + ' location')
        print('The ' + str(i + 1) + 'th object is in class ' + str(label_object_to_classify))
def main():
    print("Welcome to Dylan's Feature Selection Algorithm.")
    instances, features, data = readData()
    print(instances)
    print('This dataset has ' + str(features - 1) + ' features (not including the class attribute), with ' + str(instances) + ' instances.')
    #algorithm_type = input('Type the number of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n')
    accuracy(data, 0, 0)

main()