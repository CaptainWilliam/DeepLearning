#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from conf.segmented_func import calculate_class_for_each_point
from pybrain.datasets import ClassificationDataSet
import numpy as np

__author__ = 'LH Liu'


def get_dataset_by_func():
    """

    :return:
    """
    x1 = np.random.uniform(-3, 5, 2000)
    x2 = np.random.uniform(-3, 5, 2000)
    class_for_each_point = map(calculate_class_for_each_point, zip(x1, x2))
    return zip(x1, x2, class_for_each_point)


def split_into_train_and_test_dataset(raw_dataset, split_ratio=0.75):
    """

    :param raw_dataset:
    :param split_ratio:
    :return:
    """
    # for each test point:input(x1, x2), output(class)
    classification_dataset = ClassificationDataSet(2, 1, nb_classes=2)

    # add data element to the dataset
    for i in range(len(raw_dataset)):
        x1, x2, class_for_each_point = raw_dataset[i]
        classification_dataset.addSample([x1, x2], [class_for_each_point])

    train_dataset, test_dataset = classification_dataset.splitWithProportion(split_ratio)

    # small technique:train_dataset's out dimension change from 1 to 2, and class info in train_dataset['class']
    train_dataset._convertToOneOfMany()
    test_dataset._convertToOneOfMany()

    return train_dataset, test_dataset
