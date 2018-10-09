# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 15:56:49 2018

@author: roohollah
"""

import random
import itertools
import EMNIST_Loader
import pickle
training_data, validation_data, test_data = EMNIST_Loader.load_data_wrapper()
import network2_EMNIST

nh = [30, 50]
minibatch_size = [10, 30]
eta = [0.5, 0.8]
epochs = [2, 3]
lmbda = [50.0, 80.0]


combos=random.sample(set(itertools.product(nh,minibatch_size,eta,epochs, lmbda)), 32)
combos=sorted(combos, key=lambda tup: (tup[0],tup[1],tup[2],tup[3], tup[4]))

config=[]

savedir = 'results';

for i in range(32):
    config=combos[i][:]
    print("Number of Hidden Nodes is {}".format(config[0]))
    print("Minibatch size is {}".format(config[1]))
    print("Learning Rate (eta) is {}".format(config[2]))
    print("Number of epochs is {}".format(config[3]))
    print("Regularization parameter lmbda is {}".format(config[4]))
    
    evaluation_cost, evaluation_accuracy = [], []
    training_cost, training_accuracy = [], []
            
    net = network2_EMNIST.Network([784, config[0], 47], cost=network2_EMNIST.CrossEntropyCost) 
    net.default_weight_initializer()
    
    evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, config[3], config[1], config[2], config[4], evaluation_data=validation_data, 
            monitor_evaluation_cost=False,
            monitor_evaluation_accuracy=False,
            monitor_training_cost=False,
            monitor_training_accuracy=False)
    
    net.save(str(savedir+'/net_value_{}').format(i))
    
    f = open(str(savedir+'/run_{}.pkl').format(i), 'wb')
    pickle.dump(config,f)
    pickle.dump(evaluation_cost, f)
    pickle.dump(evaluation_accuracy, f)
    pickle.dump(training_cost, f)
    pickle.dump(training_accuracy, f)
    f.close()
     