#!/usr/bin/env python
# encoding: utf-8

import sys
import os
from math import log, ceil
from collections import defaultdict

class MinPQ:
    """This is a python version of min priority queue."""
    def __init__(self):
        self.size = 0
        self.pq = []
        if not self._check():
            print 'minpq initial check error'
    
    def push(self, i):
        if self.size == len(self.pq):
            self._resize(len(self.pq) * 2 + 1)
        self.size += 1
        self.pq[self.size - 1] = i
        self._swim()
        if not self._check():
            print 'minpq push check error'
    
    def pop(self):
        if self.size == 0:
            return
        min = self.pq[0]
        self.pq[0] = self.pq[self.size - 1]
        self.pq[self.size - 1] = float("inf")
        self.size -= 1
        self._sink()
        if self.size <= len(self.pq) / 4:
            self._resize(len(self.pq) / 2)
        return min
        if not self._check():
            print 'minpq pop check error'
    
    def min(self):
        if self.size == 0:
            return float("inf")
        return self.pq[0]
        if not self._check():
            print 'minpq min check error'
    
    def _swim(self):
        if self.size == 1:
            return
        child = self.size - 1
        parent = (child - 1) / 2
        while parent >= 0 and self.pq[child] < self.pq[parent]:
            temp = self.pq[child]
            self.pq[child] = self.pq[parent]
            self.pq[parent] = temp
            child = parent
            parent = (child - 1) / 2
        if not self._check():
            print 'minpq swim check error'
    
    def _sink(self):
        if self.size == 0 or self.size == 1:
            return
        parent = 0
        left_child = parent * 2 + 1
        right_child = parent * 2 + 2
        while parent < len(self.pq) / 2:
            mode = self._get_minimal(self.pq[parent], self.pq[left_child], self.pq[right_child])
            if mode == 0:
                break
            temp = self.pq[parent]
            if mode == 1:
                self.pq[parent] = self.pq[left_child]
                self.pq[left_child] = temp
                parent = left_child
            if mode == 2:
                self.pq[parent] = self.pq[right_child]
                self.pq[right_child] = temp
                parent = right_child
            left_child = parent * 2 + 1
            right_child = parent * 2 + 2
        if not self._check():
            print 'minpq sink check error'

    def _get_minimal(self, a, b, c):
        if a <= b and a <= c:
            return 0
        if a >= b and b <= c:
            return 1
        if a >= c and c <= b:
            return 2

    def _resize(self, new_len):
        new_pq = [float("inf")] * new_len
        for i in range(self.size):
            new_pq[i] = self.pq[i]
        self.pq = new_pq
    
    def _check(self):
        if self.size == 0 or self.size == 1:
            return True
        height = int(ceil(log(self.size, 2)))
        for i in range(2 ** (height - 1) - 1):
            if self.pq[i] > self.pq[2*i + 1] or self.pq[i] > self.pq[2*i + 2]:
                return False
        return True
