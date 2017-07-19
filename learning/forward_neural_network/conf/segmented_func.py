#!/usr/bin/env pytho
# -*- coding: utf-8 -*-
from __future__ import unicode_literals

__author__ = 'LH Liu'


def calculate_class_for_each_point(position):
    if position[0] < -1:
        return 1
    elif position[1] < -1:
        return 1
    elif 2 * position[1] + position[0] - 1 > 0:
        return 1
    else:
        return 0
