#!/usr/bin/env python
# encoding: utf-8
"""
compute_scc.py

compute strongly connected components on given graph, and output the
sizes of 5 largest sccs.
Stanford Algorithm Analysis and Design on Coursera
Created by Siqi Wu on Feb 16 2015.
"""

import sys
import os
import threading
from collections import defaultdict

def DFS1(graph, tail):
    global ft, explored, t
    explored.add(tail)
    for head in graph[tail]:
        if head not in explored:
            DFS1(graph, head)
    t += 1
    ft[tail] = t

def DFS_loop1(graph, n):
    global ft, explored, t
    ft = dict(zip(graph.keys(), [0]*n))
    explored = set([])
    t = 0
    for i in range(n, 0, -1):
        if i not in explored:
            DFS1(graph, i)
    return ft

def DFS2(graph, tail):
    global explored, t
    explored.add(tail)
    for head in graph[tail]:
        if head not in explored:
            DFS2(graph, head)
    t += 1

def DFS_loop2(graph, magical_order, n):
    global explored, t
    sizes = []
    explored = set([])
    for i in range(n, 0, -1):
        if magical_order[i] not in explored:
            t = 0
            DFS2(graph, magical_order[i])
            sizes.append(t)
    return sizes

def parse_graph(input_file):
    graph = defaultdict(list)
    graph_rev = defaultdict(list)
    for line in input_file:
        tail, head = [int(i) for i in line.rstrip().split()]
        graph[tail].append(head)
        graph_rev[head].append(tail)
    return graph, graph_rev

def main():
    input_file = open(sys.argv[1], 'r')
    graph, graph_rev = parse_graph(input_file)
    n = max([max(graph.keys()), max(graph_rev.keys())])
    magical_order = DFS_loop1(graph_rev, n)
    magical_order = {v: k for k, v in magical_order.items()}
    sccs_size = DFS_loop2(graph, magical_order, n)
    print sorted(sccs_size)[::-1][:5]

if __name__ == '__main__':
    threading.stack_size(67108864) # 64MB stack
    sys.setrecursionlimit(2 ** 20)  # approx 1 million recursions
    thread = threading.Thread(target = main) # instantiate thread object
    thread.start() # run program at target
