import numpy as np
import math
import time
#Giant Chunk of the Code was directly utilized from Proessor Eamonn's Project Debriefing
#Links here: https://www.dropbox.com/sh/0y41twimzv13ukh/AAAbkUTfSxgs6tEW9wx9G6ETa/Project_2_Briefing.mp4?dl=0
#          : https://www.dropbox.com/sh/rltooq0t3khobuj/AABg6MGQ2ysGGEF6eiI0CHq5a/Project_2_Briefing.pptx?dl=0
def readData(isMe):
    if not isMe:
        filename = input('Type in the name of the file to test: ')
    else:
        data_set = int(input("Which dataset would you like to search through? (1) for CS170_Small_Data__76 and (2) for CS170_Large_Data__87: "))
        if (data_set == 1):
            filename = 'CS170_Small_Data__76.txt'
        else:
            filename = 'CS170_Large_Data__87.txt'
    data = np.loadtxt(filename)
    #Get amount of rows and columns from our np data-structure
    #https://www.geeksforgeeks.org/find-the-number-of-rows-and-columns-of-a-given-matrix-using-numpy/
    return np.shape(data)[0], np.shape(data)[1], data

def cross_fold_accuracy(data, current_set, feature_to_add): #k cross fold algorithm in lecture
    if (feature_to_add != -1):
        current_set = current_set.copy()
        current_set.append(feature_to_add)
    number_correctly_classified = 0
    for i in range(len(data)): 
        object_to_classify = [] #Done via suggestion by a classmate to put all instances of features into its own list to lessen the hassle of manipulating data
        for features in current_set:
            object_to_classify.append(data[i][features])
        label_object_to_classify = data[i][0]
        #print('Looping over i at the ' + str(i + 1) + ' location')
        #print('The ' + str(i + 1) + 'th object is in class ' + str(label_object_to_classify))
        #How to represent infinity in Python: https://www.geeksforgeeks.org/python-infinity/
        nearest_neighbor_distance = float('inf')
        nearest_neighbor_location = float('inf')
        for k in range(len(data)):
            if k != i:
                object_to_check = [] #Done via suggestion by a classmate to put all instances of features into its own list to lessen the hassle of manipulating data
                for features in current_set:
                    object_to_check.append(data[k][features])
                #Euclidean Distance in Python with lists, other method didn't work with them https://www.geeksforgeeks.org/python-math-dist-method/
                distance = math.dist(object_to_classify, object_to_check)
                if (distance < nearest_neighbor_distance):
                    nearest_neighbor_distance = distance
                    nearest_neighbor_location = k
                    nearest_neighbor_label = data[nearest_neighbor_location][0]
        #print('Object ' + str(i + 1) + ' is class ' + str(label_object_to_classify))
        #print('Its nearest neighbor is Object ' + str(nearest_neighbor_location + 1) + ' which is in class ' + str(nearest_neighbor_label))
        if (label_object_to_classify == nearest_neighbor_label):
            number_correctly_classified += 1
    accuracy = float(number_correctly_classified) / float(len(data))
    return accuracy

def forward_search(data):
    start = time.time()
    current_set_of_features = []
    best_solution = []
    best_accuracy = 0
    for i in range(1, len(data[0])):
        #print('On the ' + str(i) + ' level of the search tree')
        feature_to_add_at_this_level = 0
        best_so_far_accuracy = 0
        for k in range(1, len(data[0])):
            if k not in current_set_of_features:
                #print('Consider adding the ' + str(k) + ' feature')
                accuracy = cross_fold_accuracy(data, current_set_of_features, k)
                print('Using feature(s) ' + str(current_set_of_features + [k]) + ' accuracy is ' + str(round(accuracy * 100, 1)) + '%')
                if accuracy > best_so_far_accuracy:
                    best_so_far_accuracy = accuracy
                    feature_to_add_at_this_level = k
        if best_so_far_accuracy >= best_accuracy: #Done to compute the best accuracy overall, not just from each feature iteration
            best_accuracy = best_so_far_accuracy
            best_solution.append(feature_to_add_at_this_level)
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        current_set_of_features.append(feature_to_add_at_this_level)
        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(round(best_so_far_accuracy * 100, 1)) + '%')
        print('\n')
        #print('Added feature ' + str(feature_to_add_at_this_level) + '. The current set is ' + str(current_set_of_features))
        #print('On level ' + str(i + 1) + ' i added feature ' + str(feature_to_add_at_this_level) + ' to current set')
    end = time.time()
    search_time = end - start
    print('Finished Search! The best feature subset was ' + str(best_solution) + ', which has an accuracy of ' + str(round(best_accuracy * 100, 1)) + '%')
    if (search_time < 60):
        print ('The duration of forward selection took ' + str(round(search_time, 2))  + ' seconds')
    elif (search_time < 3600):
        print ('The duration of forward selection took ' + str(round(search_time/60, 2))  + ' minutes')
    else:
        print ('The duration of forward selection took ' + str(round(search_time/3600, 2))  + ' hours')

def backward_search(data):
    start = time.time()
    current_set_of_features = []
    best_solution = []
    best_accuracy = 0
    for i in range(1, len(data[0])):
        current_set_of_features.append(i)
    
    for i in range(1, len(data[0]) - 1):
        #print('On the ' + str(i) + ' level of the search tree')
        feature_to_remove_at_this_level = 0
        best_so_far_accuracy = 0
        for k in current_set_of_features: #Loop goes through each feature to test removal and backwards search
            #print('Consider removing the ' + str(k) + ' feature')
            removed_feature = current_set_of_features.copy()
            removed_feature.remove(k)
            accuracy = cross_fold_accuracy(data, removed_feature, -1)
            print('Removing ' + str(k) + ' while using feature(s) ' + str(current_set_of_features) + ' accuracy is ' + str(round(accuracy * 100, 1)) + '%')
            if accuracy > best_so_far_accuracy:
                best_so_far_accuracy = accuracy
                best_solution = current_set_of_features.copy()
                feature_to_remove_at_this_level = k
        if best_so_far_accuracy >= best_accuracy: #Done to compute the best accuracy overall, not just from each feature iteration
            best_accuracy = best_so_far_accuracy
            best_solution = current_set_of_features.copy()
        else:
            print("(Warning, Accuracy has decreased! Continuing search in case of local maxima)")
        current_set_of_features.remove(feature_to_remove_at_this_level)
        print('Feature set ' + str(current_set_of_features) + ' was best, accuracy is ' + str(round(best_so_far_accuracy * 100, 1)) + '%')
        print('\n')
        #print('Removed feature ' + str(feature_to_remove_at_this_level) + '. The current set is ' + str(current_set_of_features))
        #print('On level ' + str(i + 1) + ' i removed feature ' + str(feature_to_remove_at_this_level) + ' from current set')
    end = time.time()
    search_time = end - start
    print('Finished Search! The best feature subset was ' + str(best_solution) + ', which has an accuracy of ' + str(round(best_accuracy * 100, 1)) + '%')
    if (search_time < 60):
        print ('The duration of backward elimination took ' + str(round(search_time, 2))  + ' seconds')
    elif (search_time < 3600):
        print ('The duration of backward elimination took ' + str(round(search_time/60, 2))  + ' minutes')
    else:
        print ('The duration of backward elimination took ' + str(round(search_time/3600, 2))  + ' hours')

def main():
    print("Welcome to Dylan's Feature Selection Algorithm.")
    user = int(input('Are you Dylan (1) or a regular user? (2): '))
    if (user == 1):
        isMe = True
    else:
        isMe = False
    instances, features, data = readData(isMe)
    algorithm_type = int(input('Type the number of the algorithm you want to run.\n1) Forward Selection\n2) Backward Elimination\n'))
    print('This dataset has ' + str(features - 1) + ' features (not including the class attribute), with ' + str(instances) + ' instances.')
    all_set = [1, 2, 3, 4, 5, 6]
    initial_accuracy = cross_fold_accuracy(data, all_set, -1)
    print('Running nearest neighbor with all 4 features, using "leaving-one-out" evaluation, I get an accuracy of ' + str(round(initial_accuracy * 100, 1)) + '%')
    print('Beginning search.')
    if (algorithm_type == 1):
        forward_search(data)
    else:
        backward_search(data)

main()
