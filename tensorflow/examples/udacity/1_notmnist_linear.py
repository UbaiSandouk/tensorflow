# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:53:43 2016

@author: sandouku
"""


import numpy as np
import matplotlib.pyplot as plt
from six.moves import cPickle as pickle
from sklearn import linear_model


if __name__ == "__main__":
    '''
    mypath= 'C:\\Users\\sandouku\\TensorFlow\\tensorflow\\tensorflow\\examples\\udacity\\'
    with open((mypath+'notMNIST.pickle'),'rb') as f:
        dataset = pickle.load(f)
    '''
    train_examples_number = 5000
    train_examples_filter = np.random.permutation(dataset['train_dataset'].shape[0])
    train_examples_filter = train_examples_filter[:train_examples_number]
    train_data = dataset['train_dataset'][train_examples_filter,:,:]
    train_lbls = dataset['train_labels'][train_examples_filter]
    valid_data = dataset['valid_dataset']
    valid_labels = dataset['valid_labels']
    # flaten and normalize (train, valid)
    train_data_input = train_data.reshape(train_examples_number,-1)
    valid_data_input = valid_data.reshape((valid_data.shape[0],-1))
    
    mean_vector = np.mean(train_data_input, axis=0)
    train_data_input = np.subtract(train_data_input, mean_vector)
    std_vector = np.std(train_data_input, axis=0)
    train_data_input = np.divide(train_data_input, std_vector)
    valid_data_input = np.subtract(valid_data_input, mean_vector)
    valid_data_input = np.divide(valid_data_input, std_vector)
    
    # train vector
    lr = linear_model.LogisticRegression(fit_intercept = False,n_jobs = 1, multi_class = 'multinomial', solver='lbfgs')
    lr = lr.fit(train_data_input, train_lbls)
    # evaluate
    pred_list = lr.predict(valid_data_input)
    acc = sum(valid_labels == pred_list) / float(len(pred_list))
    print "accuracy of using %d training examples is %.3f " %(train_examples_number, acc)