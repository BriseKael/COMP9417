# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 14:55:07 2017

@author: Jiarui Ding (z5045636)
@author: Quan Yin (z5042879)
"""

import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt



# Prediction Functions:

# (1) Function: prediction based on all user's ratings, return MSE
def predict_based_on_all_users (user_sim, training_set, testing_set):
    # Denominator is the sum of similarity for each user with all other users.
    denominator = np.array([np.abs(user_sim).sum(axis=1)]).T

    # Numerator is the sum of similarity of user and other users * the ratings given by other users
    numerator = user_sim.dot(training_set)

    prediction_matrix = numerator/denominator

    print('Prediction based on all users similarity is done...')
    #print(prediction_matrix)

    # Evaluation: Use the prediction and test data set to calculate MSE:

    # get the real values which are not zero in test data set.
    true_values = testing_set[testing_set.nonzero()].flatten()

    # get the predicted values of those which are not zero in test data set.
    predicted_values = prediction_matrix[testing_set.nonzero()].flatten()

    # 5.3 calculate MSE
    mse = mean_squared_error(predicted_values, true_values)

    print('The mean squared error of user_based CF is: ' + str(mse) + '\n')
    
    return mse


# (2) Function: prediction based on top-k users' ratings, return MSE:
def predict_based_on_topk_users (k, user_sim, training_set, testing_set):
    
    # Initialize a prediction matrix:   
    prediction_matrix = np.zeros(testing_set.shape)
    
    for user in range(user_sim.shape[0]):
        # exclude the get the top-k users' indexes other than user itself        
        index_of_top_k = [np.argsort(user_sim[:,user])[-2:-k-2:-1]]
    
        for item in range(training_set.shape[1]):
            
            # Denominator is the sum of similarity for each user with its top k users:
            denominator = np.sum(user_sim[user,:][index_of_top_k])
            
            # Numerator
            numerator = user_sim[user,:][index_of_top_k].dot(training_set[:,item][index_of_top_k])
            
            prediction_matrix[user, item] = numerator/denominator
            
        #print('Top-' + str(k) + ': Prediction for user ' + str(user) + '/' + str(training_set.shape[0]) + ' done...')
    
    print('Prediction based on top-' + str(k) + ' users similarity is done...')
    #print(prediction_matrix)
    
    # Evaluation: Use the prediction and test data set to calculate MSE:

    # get the real values which are not zero in test data set.
    true_values = testing_set[testing_set.nonzero()].flatten()

    # get the predicted values of those which are not zero in test data set.
    predicted_values = prediction_matrix[testing_set.nonzero()].flatten()

    # 5.3 calculate MSE
    mse = mean_squared_error(predicted_values, true_values)

    print('The mean squared error of top-' + str(k) + ' user_based CF is: ' + str(mse) + '\n')
        
    return mse



# (3) Function: give a User_ID, return the recommand the top-k Movie_ID:
def recommand_based_on_top50_users (k, uid, Rating_matrix):
    # Initialize a user sim matrix:
    user_sim = similarity(Rating_matrix)
    # Initialize a prediction matrix:   
    prediction_matrix = np.zeros(Rating_matrix.shape)
    
    # exclude the get the top-50 users' indexes other than user itself        
    index_of_top_50 = [np.argsort(user_sim[:,uid])[-2:-50-2:-1]]

    for item in range(Rating_matrix.shape[1]):
        if Rating_matrix[uid][item] == 0:
            # Denominator is the sum of similarity for each user with its top 50 users:
            denominator = np.sum(user_sim[uid,:][index_of_top_50])
            
            # Numerator
            numerator = user_sim[uid,:][index_of_top_50].dot(Rating_matrix[:,item][index_of_top_50])
            
            prediction_matrix[uid, item] = numerator/denominator
                
            #print('Top-' + str(k) + ': Prediction for user ' + str(user) + '/' + str(training_set.shape[0]) + ' done...')
    movie_ids = [i for i in np.argsort(prediction_matrix[uid, :])[-k:]]
    # return the movie id that this user has not rated
    # but his Top-50 similar user rate it high
    return movie_ids



# (4) Function: replacement of sklearn cos similarity:
def similarity(Rating_matrix):
    # sim[m ,n] = rating[m, :] X rating[n, :]
    # which is sum of movie ratings from each user u and different user u'
    # add 1e-9 make it non zero
    sim = np.dot(Rating_matrix, Rating_matrix.T) + 1e-9

    # the diagonal is just sqrt of user rating
    norms = np.array([np.sqrt(np.diagonal(sim))])
    
    return (sim / (norms * norms.T))


# (5) Function: replacement of MSE in sklearn as mean squared error:
def mean_squared_error(y_true, y_pred):
    
    return np.average((y_true - y_pred) ** 2)



# --------------Main----------------- #
if __name__ == '__main__':
    # 1. Data Load

    # 1.1 Get user-movie ratings from MovieLens('ratings.csv')
    Ratings_Names = ['User_ID', 'Movie_ID', 'Rating', 'Time_Stamp']
    df = pd.read_csv('u.data', skiprows=1, sep='\t', names=Ratings_Names)

    #print(df.head())

    # 1.2 Get user amount and item amount
    user_num = max(df.User_ID)
    item_num = max(df.Movie_ID)

    print (str(user_num) + ' Users in total.') 
    print (str(item_num) + ' Movies in total.' + '\n') 


    # 1.3.1 Initialize Rating matrix:
    Rating_matrix = np.zeros((user_num, item_num))

    # 1.3.2 Put user to movie rating into matrix 
    # UserId and MovieID starts with 0:
    num_of_entries= 0

    for entry in df.itertuples():
        Rating_matrix[entry[1]-1, entry[2]-1] = entry[3]
        num_of_entries += 1
        
    #print(Rating_matrix)

    # 2. Data process: split training and test data set:
    # We use 90% training-10% testing cross validation

    # 2.1 Calculate sparsity of the matrix
    matrix_sparsity = float(num_of_entries)
    matrix_size = user_num * item_num

    matrix_sparsity /= matrix_size
    #print (matrix_sparsity)


    # FOR EACH USER, WE TAKE 10% of matrix_sparsity*item_num as for testing
    num_items_for_each_user_test = math.floor(0.1 * matrix_sparsity * item_num)

    print(str(num_items_for_each_user_test) + ' ratings for each user are selected as testing dataset.'  + '\n') 


    # 2.2 Load Training data set and testing data set
    training_set = Rating_matrix.copy()
    testing_set = np.zeros((user_num, item_num))

    for uid in range(user_num):
        item = np.random.choice(Rating_matrix[uid, :].nonzero()[0], size=num_items_for_each_user_test, replace=False)
        
        testing_set[uid, item] = Rating_matrix[uid, item]
        training_set[uid, item] = 0.
        

    #print(testing_set)
    #print (training_set)



    # 3. Calculate similarity: using cosine_similarity provided by sklearn.metrics.pairwise
    #  User similarity
##    user_sim = similarity(training_set)
    user_sim = similarity(training_set)
    print('User based similarity matrix built...')
    #print(user_sim)

    # 4. Predict based on normal CF and Top-k based CF:
    performance = []

    # 4.1 CF based on Top-k users
    k_list = [5, 10, 50, 100, 200]


    for k in k_list:
        mse1 = predict_based_on_topk_users (k, user_sim, training_set, testing_set)
        performance.append(mse1)


    # 4.2 Normal CF:
    k_list.append(user_num)
    mse_all_user = predict_based_on_all_users (user_sim, training_set, testing_set)
    performance.append(mse_all_user)

    # 5. Visualization:
    Num_of_k_lists = (5, 10, 50, 100, 200, user_num)

    y_pos = np.arange(len(Num_of_k_lists))

     
    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, Num_of_k_lists)
    plt.ylabel('MSE')
    plt.title('Testing MSEs with varied k values')
     
    plt.show()


