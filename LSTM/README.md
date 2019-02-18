# LSTM network

A basic LSTM network built using python. It is capable of learning the arrangement of characters within a sample text and can automatically generate something that will hopefully resemble said text. It is currently designed to output a string of 200 random characters if either 1) a new low in total loss has been achieved and/or 2) if the iteration number is divisible by 500. 

The LSTM model used is mainly based off of this [blog post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) and Karpathy&#39;s [Python implementation](https://gist.github.com/karpathy/d4dee566867f8291f086) of a recurrent neural network.

### To use
Simply run the `SetUp.py` file. By default, the algorithm will be learning from a section of Mary Shelley&#39;s novel, Frakenstein, but feel free to replace it with any plain text file by changing the data entry. [Project Gutenberg](https://www.gutenberg.org/wiki/Main_Page) has a lot of free novels for use.