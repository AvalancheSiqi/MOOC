#!/usr/bin/env python
# encoding: utf-8
"""
quicksort.py

implement quicksort by 3 ways of pivots chosen.
Stanford Algorithm Analysis and Design on Coursera
Created by Siqi Wu on Feb 01 2015.
"""

import sys
import os

def quicksort_first(input_array):
    cnt = 0
    length = len(input_array)
    if length == 0 or length == 1:
        return (cnt, input_array)
    pivot = input_array[0]
    cnt += length - 1
    (i, j) = (1, 1)
    while j < length:
        if input_array[j] < pivot:
            (input_array[j], input_array[i]) = (input_array[i], input_array[j])
            i += 1
        j += 1
    (input_array[0], input_array[i-1]) = (input_array[i-1], input_array[0])
    (cnt_l, left) = quicksort_first(input_array[:i-1])
    (cnt_r, right) = quicksort_first(input_array[i:])
    return (cnt + cnt_l + cnt_r, left + [pivot] + right)

def quicksort_last(input_array):
    cnt = 0
    length = len(input_array)
    if length == 0 or length == 1:
        return (cnt, input_array)
    pivot = input_array[length-1]
    (input_array[0], input_array[length-1]) = (input_array[length-1], input_array[0])
    cnt += length - 1
    (i, j) = (1, 1)
    while j < length:
        if input_array[j] < pivot:
            (input_array[j], input_array[i]) = (input_array[i], input_array[j])
            i += 1
        j += 1
    (input_array[0], input_array[i-1]) = (input_array[i-1], input_array[0])
    (cnt_l, left) = quicksort_last(input_array[:i-1])
    (cnt_r, right) = quicksort_last(input_array[i:])
    return (cnt + cnt_l + cnt_r, left + [pivot] + right)
    
def quicksort_median(input_array):
    cnt = 0
    length = len(input_array)
    if length == 0 or length == 1:
        return (cnt, input_array)
    pivot_dict = {0: 0, 1: (length-1)/2, 2: length-1}
    array = [input_array[0], input_array[(length-1)/2], input_array[length-1]]
    pivot = sum(array) - max(array) - min(array)
    # print 'pivot is %d' % pivot
    median_place = pivot_dict[array.index(pivot)]
    (input_array[0], input_array[median_place]) = (input_array[median_place], input_array[0])
    cnt += length - 1
    (i, j) = (1, 1)
    while j < length:
        if input_array[j] < pivot:
            (input_array[j], input_array[i]) = (input_array[i], input_array[j])
            i += 1
        j += 1
    (input_array[0], input_array[i-1]) = (input_array[i-1], input_array[0])
    (cnt_l, left) = quicksort_median(input_array[:i-1])
    (cnt_r, right) = quicksort_median(input_array[i:])
    return (cnt + cnt_l + cnt_r, left + [pivot] + right)

if __name__ == "__main__":
    input_array = [int(i.rstrip()) for i in open(sys.argv[1], 'r')]
    print 'first pivot: %d' % quicksort_first(input_array[:])[0]
    print 'last pivot: %d' % quicksort_last(input_array[:])[0]
    print 'median pivot: %d' % quicksort_median(input_array[:])[0]
