import numpy as np
import os
import LSTM_P

# Plain text file
data = open('Frankestein1_to_5.txt', mode='r', encoding="utf8").read()

# Get unique characters from data
chars = set(data[:30000])

# Get number of unique chars
len_dict = len(chars)

# Create a dictionary mapping a character to a number and vice versa
c2ix = {y:x for x, y in enumerate(chars)}
ix2c = {x:y for x, y in enumerate(chars)}

# args for LSTM_P.LSTM: def __init__(self, hidden, len_dic, conv2ix, conv2ch, functions=SoftMaxCrossEntropy)
net = LSTM_P.LSTM(100, len_dict, c2ix, ix2c)

# args for update_gradient: def update_gradient(self, data, seq_length)
net.update_gradient(data[:30000], 25)
