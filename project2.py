import numpy as np
import copy
import math
def readData():
    #filename = input('Type in the name of the file to test: ')
    filename = 'CS170_Small_Data__96.txt'
    data = np.loadtxt(filename)
    #Get amount of rows and columns from our np data-structure
    #https://www.geeksforgeeks.org/find-the-number-of-rows-and-columns-of-a-given-matrix-using-numpy/
    return np.shape(data)[0], np.shape(data)[1], data

def cross_fold_accuracy(data, current_set, feature_to_add): #k cross fold algorithm in lecture
    current_set = current_set.copy()
    current_set.append(feature_to_add)
    number_correctly_classified = 0
    for i in range(len(data)):
        object_to_classify = []
        for features in current_set:
            object_to_classify.append(data[i][features])
        label_object_to_classify = data[i][0]
        #print('Looping over i at the ' + str(i + 1) + ' location')
        #print('The ' + str(i + 1) + 'th object is in class ' + str(label_object_to_classify))
        #https://www.geeksforgeeks.org/python-infinity/
        nearest_neighbor_distance = float('inf')
        nearest_neightbor_location = float('inf')
        for k in range(len(data)):
            object_to_check = []
            for features in current_set:
                object_to_check.append(data[k][features])
            if k != i:
                #print('Ask if ' + str(i + 1) + ' is nearest neighbor with ' + str(k + 1))
                #https://www.geeksforgeeks.org/python-math-dist-method/
                distance = math.dist(object_to_classify, object_to_check)
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neightbor_location = k
                    nearest_neighbor_label = data[nearest_neightbor_location][0]
        print('Object ' + str(i + 1) + ' is class ' + str(label_object_to_classify))
        print('Its nearest neighbor is Object ' + str(nearest_neightbor_location + 1) + ' which is in class ' + str(nearest_neighbor_label))
        if (label_object_to_classify == nearest_neighbor_label):
            number_correctly_classified += 1
    accuracy = number_correctly_classified/len(data)
    return accuracy

def search(data):
    rows = np.shape(data)[0]
    current_set_of_features = []
    for i in range(rows - 1):
        print('On the ' + str(i + 1) + ' level of the search tree')
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0
        for k in range(rows - 1):
            if k not in current_set_of_features:
                print('Consider adding the ' + str(k + 1) + ' feature')
                accuracy = cross_fold_accuracy(data, current_set_of_features, k + 1)
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        current_set_of_features.append(feature_to_add_at_this_level)
        print('On level ' + str(i + 1) + ' i added feature ' + str(feature_to_add_at_this_level) + ' to current set')

def main():
    print("Welcome to Dylan's Feature Selection Algorithm.")
    instances, features, data = readData()
    print(instances)
    print('This dataset has ' + str(features - 1) + ' features (not including the class attribute), with ' + str(instances) + ' instances.')
    #algorithm_type = input('Type the number of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n')
    search(data)

main()