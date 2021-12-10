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

        
        