#!/usr/bin/env python
# encoding: utf-8
"""
compute_dijkstra.py

run Dijkstra's shortest-path algorithm on graph, using 1 (the first
vertex) as the source vertex, and to compute the shortest-path distances
between 1 and every other vertex of the graph. If there is no path between
a vertex v and vertex 1, we'll define the shortest-path distance between 1
and v to be 1000000.
Stanford Algorithm Analysis and Design on Coursera
Created by Siqi Wu on Feb 21 2015.
"""

import sys
import os
from collections import defaultdict
import copy

def shortest_link(links, distance):
    min_dist = 1000000
    for k, v in links.items():
        if (v + distance[k[0]]) < min_dist:
            source = k[0]
            next = k[1]
            min_dist = v + distance[k[0]]
    return source, next, min_dist

def update(links, next, explored, nodes, graph):
    explored.add(next)
    nodes.remove(next)
    for k, v in links.items():
        if k[0] in explored and k[1] in explored:
            del links[k]
    for dst in graph[next]:
        if dst not in explored:
            links[(next, dst)] = graph[next][dst]

def dijkstra_algo(graph, nodes, src):
    distance = {}
    path = defaultdict(list)
    explored = set([])
    links = {}
    distance[src] = 0
    path[src] = []
    explored.add(src)
    nodes.remove(src)
    for dst in graph[src]:
        links[(src, dst)] = graph[src][dst]
    while(nodes != set([])):
        source, next, min_dist = shortest_link(links, distance)
        #print 'source: %s, next: %s, min_dist: %d' % (source, next, min_dist)
        distance[next] = min_dist
        temp = copy.deepcopy(path[source])
        temp.append(next)
        path[next] = temp
        update(links, next, explored, nodes, graph)
    return distance

def parse_graph(input_file):
    graph = {}
    nodes = set([])
    for line in input_file:
        adjacent = {}
        fields = line.rstrip().split()
        vertex = fields.pop(0)
        nodes.add(vertex)
        for items in fields:
            dst, dist = items.split(',')
            adjacent[dst] = int(dist)
        graph[vertex] = adjacent
    return graph, nodes

def main():
    input_file = open(sys.argv[1], 'r')
    graph, nodes = parse_graph(input_file)
    shortest_dists = dijkstra_algo(graph, nodes, '1')
    desired = [7, 37, 59, 82, 99, 115, 133, 165, 188, 197]
    output = []
    for i in desired:
        output.append(str(shortest_dists[str(i)]))
    print ','.join(output)

if __name__ == '__main__':
    main()
