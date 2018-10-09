import EMNIST_Loader
training_data, validation_data, test_data = EMNIST_Loader.load_data_wrapper()
import network2_EMNIST

import pickle

nh = 30
minibatch_size = 10
eta = 0.5
epochs = 2
lmbda = 50.0

config = [nh,minibatch_size,eta,epochs,lmbda]
print("Number of Hidden Nodes is {}".format(nh))
print("Minibatch size is {}".format(minibatch_size))
print("Learning Rate (eta) is {}".format(eta))
print("Number of epochs is {}".format(epochs))
print("Regularization parameter lmbda is {}".format(lmbda))

evaluation_cost, evaluation_accuracy = [], []
training_cost, training_accuracy = [], []
        
net = network2_EMNIST.Network([784, nh, 47], cost=network2_EMNIST.CrossEntropyCost) 
net.default_weight_initializer()

evaluation_cost, evaluation_accuracy, training_cost, training_accuracy = net.SGD(training_data, epochs, minibatch_size, eta, lmbda, evaluation_data=validation_data, 
        monitor_evaluation_cost=True,
        monitor_evaluation_accuracy=True,
        monitor_training_cost=True,
        monitor_training_accuracy=True)

net.save('net_value')

f = open('test.pkl', 'wb')
pickle.dump(config,f)
pickle.dump(evaluation_cost, f)
pickle.dump(evaluation_accuracy, f)
pickle.dump(training_cost, f)
pickle.dump(training_accuracy, f)
f.close()

f= open( "test.pkl", "rb" ) 
a1 = pickle.load(f)
a2 = pickle.load(f)
a3 = pickle.load(f)
a4 = pickle.load(f)
a5 = pickle.load(f)