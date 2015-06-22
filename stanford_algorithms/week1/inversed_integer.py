#!/usr/bin/env python
# encoding: utf-8
"""
inversed_integer.py

Stanford Algorithms Analysis and Design on Coursera
Created by Siqi Wu on Jan 23 2015.
"""

import sys
import os

def cal_inversed(input_array):
    length = len(input_array)
    if length == 1:
        return (0, input_array)
    left = input_array[:length/2]
    right = input_array[length/2:]
    (cnt_left, sorted_left) = cal_inversed(left)
    (cnt_right, sorted_right) = cal_inversed(right)
    return merge(cnt_left, cnt_right, sorted_left, sorted_right)

def merge(cnt_left, cnt_right, sorted_left, sorted_right):
    cnt = 0
    sorted_array = []

    (l_index, r_index) = (0, 0)
    while l_index < len(sorted_left) and r_index < len(sorted_right):
        if sorted_left[l_index] > sorted_right[r_index]:
            sorted_array.append(sorted_right[r_index])
            r_index += 1
            cnt += (len(sorted_left) - l_index)
        else:
            sorted_array.append(sorted_left[l_index])
            l_index += 1
    if r_index < len(sorted_right):
        sorted_array.extend(sorted_right[r_index:])
    if l_index < len(sorted_left):
        sorted_array.extend(sorted_left[l_index:])
    return (cnt + cnt_left + cnt_right, sorted_array)
    
if __name__ == '__main__':
    input_array = [int(i.rstrip()) for i in open(sys.argv[1], 'r')]
    print cal_inversed(input_array)[0]
