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

def shortest_link(links, distance):
    min_dist = 1000000
    for k, v in links.items():
        if (v + distance[k[0]]) < min_dist:
            next = k[1]
            min_dist = v + distance[k[0]]
    return next, min_dist

def update(links, next, explored, nodes, graph):
    explored.add(next)
    nodes.remove(next)
    for k, v in links.items():
        if k[0] in explored and k[1] in explored:
            del links[k]
    for n2 in graph[next].keys():
        if n2 not in explored:
            links[(next, n2)] = graph[next][n2]
    return

def dijkstra_algo(graph, nodes, src):
    distance = {}
    explored = set([])
    links = {}
    distance[src] = 0
    explored.add(src)
    nodes.remove(src)
    for n2 in graph[src].keys():
        links[(src, n2)] = graph[src][n2]
    while(nodes != set([])):
        next, min_dist = shortest_link(links, distance)
        #print 'source: %s, next: %s, min_dist: %d' % (src, next, min_dist)
        distance[next] = min_dist
        update(links, next, explored, nodes, graph)
    return distance

def parse_graph(input_file):
    graph = {}
    nodes = set([])
    for line in input_file:
        adjacent = {}
        fields = line.rstrip().split()
        n1 = int(fields.pop(0))
        nodes.add(n1)
        for items in fields:
            n2, dist = items.split(',')
            adjacent[int(n2)] = int(dist)
        graph[n1] = adjacent
    return graph, nodes

def main():
    input_file = open(sys.argv[1], 'r')
    graph, nodes = parse_graph(input_file)
    shortest_dists = dijkstra_algo(graph, nodes, 1)
    desired = [7, 37, 59, 82, 99, 115, 133, 165, 188, 197]
    output = []
    for i in desired:
        output.append(shortest_dists[i])
    print output

if __name__ == '__main__':
    main()
