#!/usr/bin/env python3

from functools import wraps
from time import time

import mnist_loader
import network

# Thanks: https://stackoverflow.com/a/27737385
def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        stop = time()
        print(f'func:{f.__name__}, args:{args} kwargs:{kwargs} took: {stop - start:0.2f} seconds.')
        return result
    return wrapper

@timing
def experiment(hidden=30, epochs=30, batch_size=10, eta=3.0):
    print(f'Training initial network with 1 hidden layer of {hidden} neurons.')
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, hidden, 10])
    net.SGD(list(training_data), epochs, batch_size, eta, test_data=list(test_data))

if __name__ == '__main__':
    experiment()
    experiment(100)
    experiment(eta=0.001)
    experiment(eta=0.01)
    experiment(eta=1)
    experiment(eta=100)
