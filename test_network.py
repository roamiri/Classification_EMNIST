# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 14:19:34 2018

@author: roohollah
"""

import random
import itertools
import EMNIST_Loader
import pickle
training_data, validation_data, test_data = EMNIST_Loader.load_data_wrapper()
import network2_EMNIST
import prettytable

nh = [30, 50]
minibatch_size = [10, 30]
eta = [0.5, 0.8]
epochs = [30, 40]
lmbda = [50.0, 80.0]


combos=random.sample(set(itertools.product(nh,minibatch_size,eta,epochs, lmbda)), 32)
combos=sorted(combos, key=lambda tup: (tup[0],tup[1],tup[2],tup[3], tup[4]))

config=[]
savedir='results'
#accuarcy = []

x = prettytable.PrettyTable(["Num Hidden Nodes" ," minibatch_size" ,"eta","epochs", "lmbda", "Test_data Accuracy"])

for i in range(32):
    config=combos[i][:]
    net = network2_EMNIST.load(str(savedir+'/net_value_{}').format(i))
    accuarcy = (100*net.accuracy(test_data)/len(test_data))
#    config_file = open(str(savedir+'/run_{}.pkl').format(i))
#    config = pickle.load(config_file)
    x.add_row([config[0],config[1],config[2],config[3],config[4], accuarcy])
    config_file.close()

print x
    