# Artificial Neural Network for Digit Recognition
Based off of Michael Nielson&#39;s &#34;Neural Networks and Deep Learning&#34; [book](http://neuralnetworksanddeeplearning.com/). The current script is a heavily modified version of the `network2.py` code, which can be found [here](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network2.py). Current modifications include allowing for matrix testing of mini batch, changing the process flow, and adding a numerical gradient check.

The script currently trains a simple artificial neural network until the end of a specified number of epochs or if an early stop is reached. At that point, it will stop the process and format the network weights and biases into a JSON object and save it as a local file. Current accuracy on the test data set sits at around 98.2%.

### How to use
Modify the network parameters in the `SetUp.py` file and run it. Note that the first layer of the network should have 784 rows, due to the input being in a 784 by 'n' matrix. For a similar reason, the final output layer should only have 10 rows. The only exception is if a numerical gradient check is being performed, in which case, the size of the input and output layers can be changed. In this instance, avoid initiating an overly large network, as it will take a long time to complete the check.
