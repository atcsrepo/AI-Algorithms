import random
import time
import json
import datetime
import math
import copy

import numpy as np

"""
This is based off of the code 'network2.py' in M. Nielson's 'Neural networks and Deep Learning' book. Note that 
there are a number of differences present.

The current code is meant to train a neural network over a defined number of epochs (unless early_stop is enabled) and 
validate it against a testing set at the end of each epoch. At the end of the program, defined by a lack of improvement 
or reaching the max number of epochs, the code will save a copy of the best weights and biases. 
"""


class QuadraticCost(object):
    @staticmethod
    def cost(opt, target):
        """
        Used to calculate training loss using quadratic cost function

        :param opt: calculated value/output
        :param target: target value
        :return: calculated loss
        """
        return 0.5 * (opt - target) ** 2

    @staticmethod
    def deltal(z, target, opt):
        """
        Used to calculate the error at the output layer
        :param z: sigmoid function
        :param target: target value
        :param opt: output value
        :return: error in the output layer
        """
        return (opt - target) * z


class CrossEntropyCost(object):
    @staticmethod
    def cost(opt, target):
        """
        Used to calculate training loss using cross entropy cost function

        :param opt: output value
        :param target: target value
        :return: cost
        """
        return np.nan_to_num(-target * np.log(opt) - (1 - target) * np.log(1 - opt))

    @staticmethod
    def deltal(z, target, opt):
        """
        Used to calculate the error at the output layer
        :param z: not used in cross entropy. Term does not exist in derivative.
        :param target: target value
        :param opt: output value
        :return: error in the output layer
        """
        return opt - target


class SigmoidFn(object):
    """
    Sigmodal activation function and 1st derivative used in back propagation
    """

    @staticmethod
    def activation(z):
        return 1 / (1 + np.exp(-z))

    def derivative(self, z):
        return self.activation(z) * (1 - self.activation(z))


class Network(object):
    def __init__(self, sizes, activation_fn=SigmoidFn, cost_fn=CrossEntropyCost,
                 monitor_costs=True, early_stop=True, half_limit=3, stop_limit=5):
        """
        :param sizes: list defining number of neurons per layer and number of layers
        :param activation_fn: name of activation function used for generating activation output
        :param cost_fn: name of cost function used to calculate training loss
        :param monitor_costs: bool specifying whether to display training costs or not while training
        :param early_stop: bool specifying whether stopping training early will be enabled
        :param half_limit: number of stagnant epochs before learning rate will be halved
        :param stop_limit: number of stagnant epochs before training is terminated, if enabled
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.monitor_costs = monitor_costs
        self.act_fn = activation_fn
        self.cost_fn = cost_fn
        self.biases = [np.random.randn(b, 1) for b in sizes[1:]]
        self.weights = self.weight_init(sizes)

        # Logs the best test accuracy as reference point
        self.current_best_acc = 0
        self.best_cost = math.inf

        # Number of rounds with no improvement before halving learning rate
        self.half_limit = half_limit
        self.stagnant_counter = 0

        self.stop_limit = stop_limit
        self.early_stop = early_stop
        self.best_biases = self.biases
        self.best_weights = self.weights

    @staticmethod
    def weight_init(sizes):
        """
        Used for initializing a weight matrix for every layer
        Note: While the weights can be initialized with:
            np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:]
        By dividing over input size, variance/stnd dev can be significantly reduced
            (e.g. N~(0, n+1 vs N~(0, 3/2))
        Which helps it train faster. Not dividing will just take more epochs to reach a similar state.

        :param sizes: an array specifying the size of each layer
        :return: an array of matrices filled with randomized weights
        """

        return [np.random.rand(y, x) / np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]

    def sgd(self, training_data, epochs, mini_batch_size, eta, test_data, lmbda=0.5):
        """
        Responsible for initiating mini batch stochastic gradient descent network training

        :param training_data: training data set
        :param epochs: max number of epochs/cycles
        :param mini_batch_size: number of inputs per mini batch
        :param eta: learning rate
        :param test_data: test data set
        :param lmbda: regularization parameter
        """
        start_time = time.time()

        train = list(training_data)
        n = len(train)

        test = list(test_data)
        n_test = len(test)

        # Train network over number of epochs, with each epoch defined by length(train)/mini_batch_size cycles
        # of training
        for j in range(epochs):
            # Shuffle and group training data into mini-batches
            random.shuffle(train)
            mini_batches = [train[k:mini_batch_size + k] for k in range(0, n, mini_batch_size)]

            # Train
            for batch in mini_batches:
                self.update_grad(batch, eta, lmbda, n)

            # Test for training accuracy on test dataset and get training costs for training set
            test_acc = self.evaluate_accuracy(test)
            training_costs = self.tracked_cost(train, lmbda)
            print('Epoch {0}: {1} / {2} correctly id\'ed in test data set'.format(j, test_acc, n_test))

            # If cost monitoring is enabled (in __init__), shows current cost vs best cost.

            if self.monitor_costs:
                print('Average training cost was: {0:.5f} vs best cost {1:.5f}'.format(training_costs, self.best_cost))

            """
            Check for improvements in costs. Add to counter if no improvement; otherwise, reset counter.
            Improvement defined by >1% decrease in current training cost vs best cost  
            """

            if self.best_cost * .99 > training_costs:
                self.stagnant_counter = 0
                self.best_cost = training_costs
                print('Stagnant counter: {0}'.format(self.stagnant_counter))
            else:
                self.stagnant_counter += 1
                print('Stagnant counter: {0}'.format(self.stagnant_counter))

            # If early stop enabled (in __init__), break out of for loop if no improvement observed
            # after 'early_stop' number of epochs
            if self.early_stop:
                if self.stagnant_counter == self.stop_limit:
                    print('No improvement has been observed in 5 epochs. Stopping now.')
                    break

            # Halves learning rate if no improvement observed in 'half_limit' number of epochs.
            if self.stagnant_counter == self.half_limit:
                eta /= 2
                print('***Learning rate decreased***')

            print('Time elapsed: {}'.format((time.time() - start_time)))
            print('--------------------------------')

        # Save the data into a local file
        self.save(datetime.datetime.now().strftime("%Y-%m-%d") + " Network parameters")

    def update_grad(self, data, eta, lmbda, n):
        """
        Update the network's weights and biases by applying gradient descent using back-propagation on a single
         mini batch of input.

        :param data: mini batch of training set
        :param eta: learning rate
        :param lmbda: regularization parameter
        :param n: data set size
        """

        # The original code calculates the change in nabla_w and nabla_b (delta_nabla_w and delta_nabla_b respectively)
        # for each individual input and then accumulates it using a for loop
        # Here we convert it into a 784 by mini_batch_size matrix and calculate it all in one go
        x = np.asarray([np.array(x.ravel()) for x, y in data]).transpose()  # inputs
        y = np.asarray([np.array(y.ravel()) for x, y in data]).transpose()  # targets

        # Performs the forward and back propgation for the data
        nabla_b, nabla_w = self.backprop(x, y)

        # Update weights and biases
        self.biases = [b - (eta / len(data)) * nb for b, nb in zip(self.biases, nabla_b)]
        self.weights = [(1 - eta * (lmbda / n)) * w - (eta / len(data)) * nw for w, nw in zip(self.weights, nabla_w)]

    def backprop(self, ipt, opt):
        """

        :param ipt: array of input values
        :param opt: array of target values
        :return: an array of array/matrices for nabla_w and nabla_b (dC/dw and dC/db)
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = ipt
        activations = [ipt]  # Store list of activations, layer by layer
        zs = []  # Store list of z vectors, layer by layer

        # Similar to feed forward, except with data storage
        for b, w in zip(self.biases, self.weights):
            # Calculate z-values for a given layer and store it
            z = np.dot(w, activation) + b
            zs.append(z)

            # Calculate activation from z-values for a given layer and store it
            activation = self.act_fn.activation(z)
            activations.append(activation)

        # Calculate cost at the output layer (dCost/dActivation)
        delta_cost = self.cost_fn.deltal(self.act_fn.derivative(self.act_fn, zs[-1]), opt, activations[-1])

        # Setting up for the actual back-propagation
        # Because our input is a matrix, delta_cost is also a matrix; however, this will not work for creating nabla_b
        # so we will sum down columns and reshape accordingly
        nabla_b[-1] = np.sum(delta_cost, axis=1).reshape(nabla_b[-1].shape)
        nabla_w[-1] = np.dot(delta_cost, activations[-2].transpose())

        # Back-propagation sequence
        for l in range(2, self.num_layers):
            sp = self.act_fn.derivative(self.act_fn, zs[-l])  # First derivative of activation fcn
            delta_cost = np.dot(self.weights[-l + 1].transpose(), delta_cost) * sp
            nabla_b[-l] = np.sum(delta_cost, axis=1).reshape(nabla_b[-l].shape)  # Note letter 'l' vs number '1'
            nabla_w[-l][:] = np.dot(delta_cost, activations[-l - 1].transpose())

        return nabla_b, nabla_w

    def evaluate_accuracy(self, data):
        """
        Determines the number of correctly identified hand written digits within the test (or validation) data set

        This is not meant to be used with the training data set, which is encoded differently.
        Refer to the minst_loader.py or the original 'network2.py' file to see how conversion is handled.

        :param data: test or validation data set, which consists of tuples (x, y), where 'x' is a 784-dimensional
            representation of the digit and 'y' is a 10-dimensional vector representing the digit
        :return: the total number of correctly classified results
        """

        results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)

    def feedforward(self, a, weights=None):
        """
        Performs the forward propagation step only.

        :param a: vector/matrix representation of the input data
        :param weights: Doesn't really have a purpose currently, as weights are drawn from the instance itself. However,
            it can be adopted so that it can be used with a pre-defined network.
        :return: a vector representing the final activations
        """
        for b, w in zip(self.biases, weights or self.weights):
            a = self.act_fn.activation(np.dot(w, a) + b)
        return a

    def tracked_cost(self, data, lmbda):
        """
        Used to determine the cost/lost associated with training.

        :param data: refers to training data set
        :param lmbda: regularization parameter
        :return: total cost associated with training
        """
        cost = 0

        # First calculate the cost derived from cost function itself
        for x, y in data:
            cost += self.cost_fn.cost(self.feedforward(x), y).sum() / len(data)

        # Append cost associated with L2 regularization to sub-total cost
        cost += 0.5 * (lmbda / len(data)) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost

    def save(self, filename):
        """
        Save the neural network to the file ``filename``.

        :param filename: file name for saved file
        """
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.best_weights],
                "biases": [b.tolist() for b in self.best_biases],
                "cost": str(self.best_cost)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()

        print('Network details saved at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    def num_gradient_check(self, ipt, opt):
        """
        Since the gradient dc/db is dependent only on error of associated layer, we will ignore it, as dC/dw is
        dependent on error, in addition to activation. Hence, if dC/dw is correct, dC/db will most likely be as well.
        """
        eps = .0001

        num_nabla_w = [np.zeros(w.shape) for w in self.weights]

        for m in range(len(self.weights)):
            for r in range(self.weights[m].shape[0]):
                for c in range(self.weights[m].shape[1]):
                    max_matrix = copy.deepcopy(self.weights)
                    max_matrix[m][r][c] += eps

                    min_matrix = copy.deepcopy(self.weights)
                    min_matrix[m][r][c] -= eps

                    num_grad = (self.cost_fn.cost(self.feedforward(ipt, weights=max_matrix), opt).sum() -
                                self.cost_fn.cost(self.feedforward(ipt, weights=min_matrix), opt).sum()) / (2 * eps)

                    num_nabla_w[m][r][c] = num_grad

        _, nabla_w = self.backprop(ipt, opt)

        for n in range(len(self.weights)):
            difference = abs(nabla_w[n] - num_nabla_w[n]) / abs(nabla_w[n] + num_nabla_w[n])
            print("Layer: {}".format(n), difference)
