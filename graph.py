import pandas as pd
import numpy as np


class Graph:
    
    def __init__(self, name = ""):
        self.name = name
        self.nodes = {}
        self.neighbors = {}
        self.edges = []
        return
    
    def __getitem__(self, node):
        return self.nodes[str(node)]
    
    def size(self):
        return len(self.nodes)
    
    def summary(self):
        print(f"{self.size()} nodes")
        print(f"{len(self.edges)} edges")
        return
    
    def add_node(self, node):   
        node = str(node)
        
        self.nodes[node] = {}
        self.neighbors[node] = set()
        
        return
    
    def add_edge(self, src, tgt, prop):
        
        src = str(src)
        tgt = str(tgt)
        
        try:
            self.neighbors[src].add(tgt)
        except KeyError:
            self.add_node(src)
            self.neighbors[src].add(tgt)
            
        try:
            self.neighbors[tgt].add(src) 
        except KeyError:
            self.add_node(tgt)
            self.neighbors[tgt].add(src)

        self.nodes[src][tgt] = prop
        
        self.edges.append((src, tgt))
        
        return


def merge_dicts(a, b, subkey):
    
    if not a:
        return b
    
    if not b:
        return a
    
    c = a
        
    for k in b.keys():
        try:
            if b[k][subkey] < c[k][subkey]:
                c[k] = {subkey: b[k][subkey]}
        except KeyError:
            c[k] = {subkey: b[k][subkey]}

    return c
    

def merge_graphs(G, J, assign_weights = True):
    
    new_graph = Graph()
    
    # all unique nodes in both graphs
    all_nodes = set(list(G.nodes.keys()) + list(J.nodes.keys()))
    
    for node in all_nodes:
        
        # find all edges in both graphs for that node
        try:
            G_node_edges = G[node]
        except KeyError:
            G_node_edges = {}
            
        try:
            J_node_edges = J[node]
        except KeyError:
            J_node_edges = {}
            
        # merge them so we get a unique dictionary
        # for edges in common only that with the earlier timestamp
        # is kept
        node_edges = merge_dicts(G_node_edges, J_node_edges, "time")
           
        # add those edges to the new graph
        for edge in node_edges:
            new_graph.add_edge(node, edge, 
                               {"time": node_edges[edge]["time"]})
    
    return new_graph    
        