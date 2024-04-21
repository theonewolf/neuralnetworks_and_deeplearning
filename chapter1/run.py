#!/usr/bin/env python3

import mnist_loader
import network

if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(list(training_data), 30, 10, 3.0, test_data=list(test_data))
