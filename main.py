#!/usr/bin/env python

from __future__ import division


import argparse

from echo_state_network.ESN import ESN


#from echo_state_network.ESN import ESN


def parse_options():
    optparser = argparse.ArgumentParser(description='Echo State Network.')
    optparser.add_argument(
        '-f', '--input_file',
        dest='filename',
        help='filename containing data',
        default='datasets/mnist.pkl.gz'
    )
    optparser.add_argument(
        '-s', '--sparsity',
        dest='sparsity',
        help='sparsity of Reservoir',
        default=0.2,
        type=float
    )
    optparser.add_argument(
        '-n', '--neurons',
        dest='neurons',
        help='number of neurons in Reservoir',
        default=100,
        type=int
    )
    optparser.add_argument(
        '-p', '--instances',
        dest='instances',
        help='number of instances to train on',
        default=None,
        type=int
    )
    optparser.add_argument(
        '-alfa', '--alfa',
        dest='alfa',
        help='reservoir matrix scaling parametr',
        default=0.5,
        type=float
    )
    optparser.add_argument(
        '-r', '--r_principal_component',
        dest='principal_components',
        help='first R principal components',
        default=None,
        type=int
    )
    optparser.add_argument(
        '-w', '--washout',
        dest='washout',
        help='washout period',
        default=4,
        type=int
    )
    return optparser.parse_args()


def run_esn_regularized_least_squares(options):
    network = ESN(options.filename, options.neurons, options.alfa, options.sparsity, options.principal_components, options.washout)
    network.load_dataset()
    network.initialize()
    network.train_for_regularized_least_squares()
    accuracy = network.classify_for_regularized_least_squares()
    return accuracy


def run_esn_clustering_with_principal_components_approach_1(options):
    network = ESN(options.filename, options.neurons, options.alfa, options.sparsity, options.principal_components, options.washout)
    network.load_dataset()
    network.initialize()
    network.train_for_clustering_with_principal_components_approach1()
    accuracy = network.classify_for_clustering_with_principal_components_approach1()
    return accuracy


def run_esn_clustering_with_principal_components_approach_2(options):
    network = ESN(options.filename, options.neurons, options.alfa, options.sparsity, options.principal_components, options.washout)
    network.load_dataset()
    network.initialize()
    network.train_for_clustering_with_principal_components_approach2_paralel()
    accuracy = network.classify_for_clustering_with_principal_components_approach2()
    return accuracy


def run_esn_clustering_with_principal_components_approach_3(options):
    network = ESN(options.filename, options.neurons, options.alfa, options.sparsity, options.principal_components, options.washout)
    network.load_dataset()
    network.initialize()
    network.train_for_clustering_with_principal_components_approach3_paralel()
    accuracy = network.classify_for_clustering_with_principal_components_approach3()
    return accuracy


options = parse_options()
#accuracy = run_esn_regularized_least_squares(options)
#accuracy = run_esn_clustering_with_principal_components_approach_1(options)
accuracy = run_esn_clustering_with_principal_components_approach_2(options)
#accuracy = run_esn_clustering_with_principal_components_approach_3(options)
print '\033[92mAccuracy: {0} \033[0m'.format(accuracy)
