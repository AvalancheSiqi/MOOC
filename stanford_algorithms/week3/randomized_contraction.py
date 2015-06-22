#!/usr/bin/env python
# encoding: utf-8
"""
randomized_contraction.py

implement randomized contraction algorithm for the min cut problem.
Stanford Algorithm Analysis and Design on Coursera
Created by Siqi Wu on Feb 07 2015.
"""

import sys
import os
import random
import math

class UnionFind:
    """This is an union find API of python."""
    def __init__(self, array):
        self.length = len(array)
        self.id = [0]*self.length
        for i in range(self.length):
            self.id[i] = i+1
    def union(self, a, b):
        if self.connected(a, b):
            return
        aid = self.id[a-1]
        bid = self.id[b-1]
        for i in range(self.length):
            if self.id[i] == bid:
                self.id[i] = aid
    def find(self, a):
        return self.id[a-1]
    def connected(self, a, b):
        return self.id[a-1] == self.id[b-1]

def parse_graph(input_file):
    vertices = []
    edges = set([])
    for line in input_file:
        fields = [int(i) for i in line.rstrip().split()]
        vertex = fields.pop(0)
        vertices.append(vertex)
        edge = [tuple(sorted([vertex, i])) for i in fields]
        edges.update(edge)
    return vertices, list(edges)

def randomized_contraction(vertices, edges):
    n = len(vertices)
    graph = UnionFind(vertices)
    for t in range(n-2):
        (a, b) = random.choice(edges)
        while graph.connected(a, b):
            (a, b) = random.choice(edges)
        graph.union(a, b)
    result = 0
    for edge in edges:
        if not graph.connected(edge[0], edge[1]):
            result += 1
    return result

if __name__ == '__main__':
    input_file = open(sys.argv[1], 'r')
    vertices, edges = parse_graph(input_file)
    cuts = []
    n = len(vertices)
    for t in range(1000):
    #for t in range(n*n*int(math.sqrt(n))):
        cut = randomized_contraction(vertices[:], edges[:])
        cuts.append(cut)
    print 'min cut is %d' % min(cuts)
