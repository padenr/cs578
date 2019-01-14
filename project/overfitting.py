"""

    overfitting

    ~~~~~~~~~~~

    Plot graphs to illustrate the problem of overfitting.

    """



# Standard library

import json

import random

import sys

import argparse



# My library

sys.path.append('../src/')

import mnist_loader

import network2

from network2 import linear_up
from network2 import linear_down
from network2 import quadratic
from network2 import sigmoid_scaled
from network2 import sigmoid_inverse



# Third-party libraries

import matplotlib.pyplot as plt

import numpy as np


def main(filename, num_epochs,

         training_cost_xmin=200,

         test_accuracy_xmin=200,

         test_cost_xmin=0,

         training_accuracy_xmin=0,

         training_set_size=1000,

         lmbda=0.0,
         
         which_lmbda="l1",
         
         dropout=False,
         
         func=None,
         
         func_str = None):

    """``filename`` is the name of the file where the results will be

        stored.  ``num_epochs`` is the number of epochs to train for.

        ``training_set_size`` is the number of images to train on.

        ``lmbda`` is the regularization parameter.  The other parameters

        set the epochs at which to start plotting on the x axis.

        """

    run_network(filename, num_epochs, training_set_size, lmbda, which_lmbda, dropout, func)

    make_plots(filename, num_epochs,

               training_cost_xmin,

               test_accuracy_xmin,

               test_cost_xmin,

               training_accuracy_xmin,

               training_set_size,
               
               func_str)



def run_network(filename, num_epochs, training_set_size=1000, lmbda=0.0, which_lmbda="l1", dropout=False, func=None):

    """Train the network for ``num_epochs`` on ``training_set_size``

        images, and store the results in ``filename``.  Those results can

        later be used by ``make_plots``.  Note that the results are stored

        to disk in large part because it's convenient not to have to

        ``run_network`` each time we want to make a plot (it's slow).

        """

    # Make results more easily reproducible

    random.seed(12345678)

    np.random.seed(12345678)

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

    net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost())

    net.large_weight_initializer()

    test_cost, test_accuracy, training_cost, training_accuracy = net.SGD(training_data[:training_set_size], num_epochs, 10, 0.5,

                  which_lmbda=which_lmbda,

                  dropout=dropout,

                  func=func,

                  evaluation_data=test_data, lmbda = lmbda,

                  monitor_evaluation_cost=True,

                  monitor_evaluation_accuracy=True,

                  monitor_training_cost=True,

                  monitor_training_accuracy=True)

    f = open(filename, "w")

    json.dump([test_cost, test_accuracy, training_cost, training_accuracy], f)

    f.close()



def make_plots(filename, num_epochs,

               training_cost_xmin=200,

               test_accuracy_xmin=200,

               test_cost_xmin=0,

               training_accuracy_xmin=0,

               training_set_size=1000,
               
               func_str = "None"):

    """Load the results from ``filename``, and generate the corresponding

        plots. """

    f = open(filename, "r")

    test_cost, test_accuracy, training_cost, training_accuracy = json.load(f)

    f.close()

    #plot_training_cost(training_cost, num_epochs, training_cost_xmin)

    #plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin)

    #plot_test_cost(test_cost, num_epochs, test_cost_xmin)

    #plot_training_accuracy(training_accuracy, num_epochs,

    #                       training_accuracy_xmin, training_set_size)

    plot_overlay(test_accuracy, training_accuracy, num_epochs,

             min(test_accuracy_xmin, training_accuracy_xmin),

             training_set_size, lmbda, func_str)



def plot_training_cost(training_cost, num_epochs, training_cost_xmin):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(training_cost_xmin, num_epochs),  training_cost[training_cost_xmin:num_epochs], color='#2A6EA6')

    ax.set_xlim([training_cost_xmin, num_epochs])

    ax.grid(True)

    ax.set_xlabel('Epoch')

    ax.set_title('Cost on the training data')

    plt.show()



def plot_test_accuracy(test_accuracy, num_epochs, test_accuracy_xmin):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(test_accuracy_xmin, num_epochs),

            [accuracy/100.0

             for accuracy in test_accuracy[test_accuracy_xmin:num_epochs]],

            color='#2A6EA6')

    ax.set_xlim([test_accuracy_xmin, num_epochs])

    ax.grid(True)

    ax.set_xlabel('Epoch')

    ax.set_title('Accuracy (%) on the test data')

    plt.show()



def plot_test_cost(test_cost, num_epochs, test_cost_xmin):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(test_cost_xmin, num_epochs),

            test_cost[test_cost_xmin:num_epochs],

            color='#2A6EA6')

    ax.set_xlim([test_cost_xmin, num_epochs])

    ax.grid(True)

    ax.set_xlabel('Epoch')

    ax.set_title('Cost on the test data')

    plt.show()



def plot_training_accuracy(training_accuracy, num_epochs,

                           training_accuracy_xmin, training_set_size):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    ax.plot(np.arange(training_accuracy_xmin, num_epochs),

            [accuracy*100.0/training_set_size

             for accuracy in training_accuracy[training_accuracy_xmin:num_epochs]],

            color='#2A6EA6')

    ax.set_xlim([training_accuracy_xmin, num_epochs])

    ax.grid(True)

    ax.set_xlabel('Epoch')

    ax.set_title('Accuracy (%) on the training data')

    plt.show()



def plot_overlay(test_accuracy, training_accuracy, num_epochs, xmin,

                 training_set_size, lmbda, func_str):

    fig = plt.figure()

    ax = fig.add_subplot(111)

    if func_str == "linup":
        func_str = "Linear - Positive: "
    elif func_str == "lindown":
        func_str = "Linear - Negative: "
    elif func_str == "quadratic":
        func_str = "Quadratic: "
    elif func_str == "sig":
        func_str = "Sigmoid 1 to 0: "
    elif func_str == "sigin": 
        func_str = "Sigmoid 0 to 1: "

    ax.set_title("Dropout")

    ax.plot(np.arange(xmin, num_epochs),

            [accuracy/100.0 for accuracy in test_accuracy],

            color='#2A6EA6',

            label="Accuracy on the test data")

    ax.plot(np.arange(xmin, num_epochs),

            [accuracy*100.0/training_set_size

                for accuracy in training_accuracy],

            color='#FFA933',

            label="Accuracy on the training data")

    ax.grid(True)

    ax.set_xlim([xmin, num_epochs])

    ax.set_xlabel('Epoch')

    ax.set_ylabel('Accuracy (%)')

    ax.set_ylim([90, 100])

    plt.legend(loc="lower right")

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--func", help="which function over time?")
    parser.add_argument("--reg", help="l1 or l2")
    parser.add_argument("--dropout", help="y or n")
    parser.add_argument("--lmbda", help="this is the lambda value")
    args = parser.parse_args()

    if args.reg:
        r_type = args.reg
    else:
        print "please pick a regularization"
        exit()

    if args.func:
        func_str = args.func
    else:
        func_str = None

    if args.func == "linup":
        func = linear_up
    elif args.func == "lindown":
        func = linear_down
    elif args.func == "sig":
        func = sigmoid_scaled
    elif args.func == "sigin":
        func = sigmoid_inverse
    elif args.func == "quadratic":
        func = quadratic
    else:
        func = None

    if args.lmbda:
       lmbda = float( args.lmbda ) #For experiment 1
    else:
        lmbda = 0 ## for experiment 2 this is set by the internal function now

    if args.dropout:
        dropout = True
    else:
        dropout = False

    print lmbda
    filename =  "results.txt" #raw_input("Enter a file name: ")

    num_epochs = 30 #int(raw_input("Enter the number of epochs to run for: "))
    training_cost_xmin = 0 #int(raw_input("training_cost_xmin (suggest 200): "))
    test_accuracy_xmin = 0 #int(raw_input("test_accuracy_xmin (suggest 200): "))
    test_cost_xmin = 0 # int(raw_input("test_cost_xmin (suggest 0): "))
    training_accuracy_xmin = 0 #int(raw_input("training_accuracy_xmin (suggest 0): "))
    training_set_size = 50000 #int(raw_input("Training set size (suggest 1000): "))
    main(filename, num_epochs, training_cost_xmin,
    test_accuracy_xmin, test_cost_xmin, training_accuracy_xmin,training_set_size, lmbda, r_type, dropout, func, func_str)

