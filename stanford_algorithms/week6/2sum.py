#!/usr/bin/env python
# encoding: utf-8
"""
2sum.py

Compute the number of target values t in the interval [-10000,10000]
(inclusive) such that there are distinct numbers x,y in the input file
that satisfy x+y=t.
Stanford Algorithm Analysis and Design on Coursera
Created by Siqi Wu on Mar 02 2015.
"""

import sys
import os
from collections import defaultdict

def hash_func(array, r):
    dict = defaultdict(list)
    for i in array:
        k = i / r
        dict[k].append(i)
    return dict

def two_sum(array, lo, hi):
    w = set([])
    r = hi - lo + 1
    n = len(array)
    left = 0
    right = n - 1
    d = hash_func(array, r)
    while (left < right):
        num = array[left] + array[right]
        if num > hi:
            right -= 1
        elif num < lo:
            left += 1
        else:
            while (left < right):
                sd1 = d[(lo - array[left]) / r]
                sd2 = d[(lo - array[left]) / r + 1]
                search_domain = sd1
                search_domain.extend(sd2)
                for b in search_domain:
                    num = array[left] + b
                    if array[left] != b and num >= lo and num <= hi:
                        w.add(num)
                left += 1
            break
    return len(w)

if __name__ == '__main__':
    array = [int(i.rstrip()) for i in open(sys.argv[1], 'r')]
    s = set(array)
    array = list(s)
    lo = -10000
    hi = 10000
    array.sort()
    result = two_sum(array, lo, hi)
    print result
