#!/usr/bin/env python
# encoding: utf-8
"""
median_maintenance.py

Compute the sum of these 10000 medians, modulo 10000,
treat input as a stream of numbers, arriving one by one.
Stanford Algorithm Analysis and Design on Coursera
Created by Siqi Wu on Mar 05 2015.
"""

import sys
import os
import minpq
import maxpq

if __name__ == '__main__':
    input_file = open(sys.argv[1], 'r')
    sum = 0
    smaller = maxpq.MaxPQ()
    larger = minpq.MinPQ()
    smaller.push(int(input_file.readline()))
    sum += smaller.max()
    for i in input_file:
        i = int(i.rstrip())
        if smaller.size == larger.size:
            if i <= larger.min():
                smaller.push(i)
            else:
                temp = larger.pop()
                larger.push(i)
                smaller.push(temp)
        else:
            if i >= smaller.max():
                larger.push(i)
            else:
                temp = smaller.pop()
                smaller.push(i)
                larger.push(temp)
        sum += smaller.max()
    print sum % 10000
