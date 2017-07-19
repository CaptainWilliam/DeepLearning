#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from utils.output_question import output_params_info
from pre_process import get_dataset_by_func, split_into_train_and_test_dataset
from neural_network_model import create_neural_network_model, create_default_neural_network_model, train_model_by_bp

__author__ = 'LH Liu'


def process():
    """

    :return:
    """
    # 1.get random train and test dataset from figure.1
    raw_data = get_dataset_by_func()
    train_dataset, test_dataset = split_into_train_and_test_dataset(raw_data, split_ratio=0.75)

    # 2.get neural network using pybrain
    fnn = create_default_neural_network_model(train_dataset)
    # fnn = create_neural_network_model(train_dataset)
    test_error = train_model_by_bp(fnn, train_dataset, test_dataset)
    print '\nModel Precision: {}\n'.format((100-test_error) * 0.01)

    # 3.output question's answer
    output_params_info(fnn)
