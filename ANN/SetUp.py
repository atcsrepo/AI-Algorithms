import mnist_loader
import Matrix_ANN

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

"""
def __init__(self, sizes, activation_fn=SigmoidFn, cost_fn=CrossEntropyCost,
                 monitor_costs=True, early_stop=True, half_limit=3, stop_limit=5)
                 
                 
sgd(self, training_data, epochs, mini_batch_size, eta, lmbda=0.0, test_data=None),where:
    training_data: training set data
    epochs: maximum number of training rounds
    mini_batch_size: size of batch used to estimate gradient vector
    eta: initial learning rate
    lmbda: regularization parameter
    test_data: testing set data
"""

# first layer should always be 784 due to nature of input
net = Matrix_ANN.Network([784, 100, 100, 10])
net.sgd(training_data, 100, 50, 1, test_data, lmbda=5)

"""
# used for gradient check; otherwise unneeded
import numpy as np

ipt = np.reshape([0.5 for x in range(0,50)], (50, 1))

opt = np.reshape([0 for x in range(0,10)], (10, 1))
opt[2][0] = 1

# avoid large networks
net = Matrix_ANN.Network([50, 20, 20, 10])
net.num_gradient_check(ipt, opt)
"""