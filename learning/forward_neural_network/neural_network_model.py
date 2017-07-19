#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork
from pybrain.utilities import percentError
from pybrain.structure import *
import logging

__author__ = 'LH Liu'

logger = logging.getLogger()


def create_default_neural_network_model(train_dataset):
    return buildNetwork(train_dataset.indim, 10, 5, train_dataset.outdim, outclass=SoftmaxLayer)


def create_neural_network_model(train_dataset):

    # also could use shortcut buildNetwork func
    fnn = FeedForwardNetwork()

    # create three layers,
    in_layer = LinearLayer(train_dataset.indim, name='inLayer')
    hidden_layer = SigmoidLayer(10, name='hiddenLayer')
    out_layer = LinearLayer(train_dataset.outdim, name='outLayer')

    # add three layers to the neural network
    fnn.addInputModule(in_layer)
    fnn.addModule(hidden_layer)
    fnn.addOutputModule(out_layer)

    # link three layers
    in_to_hidden = FullConnection(in_layer, hidden_layer)
    hidden_to_out = FullConnection(hidden_layer, out_layer)

    # add the links to neural network
    fnn.addConnection(in_to_hidden)
    fnn.addConnection(hidden_to_out)

    # make neural network come into effect
    fnn.sortModules()

    return fnn


def train_model_by_bp(fnn, train_dataset, test_dataset):
    # train the NN
    trainer = BackpropTrainer(fnn, train_dataset, verbose=True, weightdecay=0.01)
    # set the epoch times to make the NN  fit
    trainer.trainUntilConvergence(maxEpochs=50)
    test_result = percentError(trainer.testOnClassData(dataset=test_dataset), test_dataset['class'])

    if test_result > 15:
        logger.info('Test Result\'s correct ratio is below 85%')

    return test_result
