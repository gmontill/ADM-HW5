import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    """
    Implementation of a directed graph
    """
    
    def __init__(self, name = ""):  
        # name of the graph
        self.name = name
        
        # dictionary with the nodes of the graph as keys
        # and the dictionary with the target nodes of each as values;
        # each of those dictionaries (the edges) has an attribute "time"
        # with the timestamp of that edge and an attribute "weight"
        # with its weight
        self.nodes = {}
        
        # dictionary with the nodes of the graph as keys
        # and the neighboring nodes of each as values
        self.neighbors = {}
        
        # list of the pairs (source node, target node)
        self.edges = []
        
        return
    
    def __getitem__(self, label):
        """
        Returns the node with label 'label',
        i.e. the dictionary containing all of
        its target nodes and timestamps and weights
        of each of those edges
        """
        
        return self.nodes[str(label)]
    
    def summary(self):
        """
        Prints a summary of the graph 
        with the total number of nodes and edges
        """
        
        print(f"Graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
    
    def add_node(self, label):
        """
        Adds a node to the graph
        
        Arguments
            label : (int), (str) or whatever other object
            
        Returns
            None
        """
        
        label = str(label)
        
        self.nodes[label] = {}
        self.neighbors[label] = set()
        
        return
    
    def add_edge(self, src, tgt):
        """
        Adds an edge between a source and a target node;
        if one or both nodes are not already in the graph
        they will be added;
        if there is already an edge between two nodes,
        then the one going the opposite way is discarded
        (i.e. the edge added before has priority)
        
        Arguments
            src, tgt : (int), (str) or whatever other object
            
        Returns
            None
        """
        
        src = str(src)
        tgt = str(tgt)
        
        # if there is already an edge (tgt -> src)
        # then don't add this one (src –> tgt)
        try:
            if self.nodes[tgt][src]:
                return
        except:
            pass
        
        # if the src node already exists in the graph
        # then add tgt to its neighbors,
        # otherwise create it first
        try:
            self.neighbors[src].add(tgt)
        except KeyError:
            self.add_node(src)
            self.neighbors[src].add(tgt)
            
        # if the tgt node already exists in the graph
        # then add src to its neighbors,
        # otherwise create it first
        try:
            self.neighbors[tgt].add(src) 
        except KeyError:
            self.add_node(tgt)
            self.neighbors[tgt].add(src)

        # create an edge with direction
        # (src —> tgt)
        self.nodes[src][tgt] = {}
        
        self.edges.append((src, tgt))
        
        return

    
def test_graph(G, Gx):
    """
    Checks whether the graph 'G' built 
    with this implementation and the graph 'Gx'
    built leveraging 'networkx' are effectively 
    the same, i.e. they have the same nodes and
    each node has the same neighbors
    
    Remark: it assumes every node is labelled as a string
    
    Arguments
        G  : graph built with this module 
        Gx : graph built with networkx
    
    Returns
        None
    """
    
    # same number of nodes?
    if set(G.nodes.keys()).difference(set(Gx.nodes)):
        print("Bad graph! Too few/many nodes")
        return
    
    # same neighbors for each node?
    for node in G.nodes.keys():
        node = str(node)
        if G.neighbors[node].difference(set(Gx.neighbors(node))):
            print(f"Bad graph! Too few/many neighbors for node '{node}'")
            return

    print("OK! Same nodes and neighbors")
    
    return


def graph_from_df(df, kind = "custom"):
    """
    Fills a graph from a dataframe
    with a 'src' column containing
    all starting nodes, a 'tgt' column
    containing all target nodes and
    a 'timestamp' column with the timestamp
    of that link; each of these edges is 
    also assigned a weight of 1
    
    Arguments
        df : pandas dataframe
    
    Returns
        a graph G with all the nodes and edges in df
    """
    
    if kind == "nx" or kind == "networkx":
        G = nx.Graph()
    else:
        G = Graph()
    
    for idx in df.index: 
        row = df.iloc[idx]
        
        src = str(row["src"])
        tgt = str(row["tgt"])
        
        if src != tgt:
            
            G.add_edge(src, tgt)
            
            try:
                G[src][tgt]["time"] = row["timestamp"]
                G[src][tgt]["weight"] = 1
            except KeyError:
                pass
                
        
    return G


def merge_edges(a, b):
    """
    Merges the two dictionaries each containing
    the target nodes in the two graphs 
    of the same source node
    (i.e. given a source node 'u', 
    'a' contains the target nodes of 'u' in graph G_a,
    and 'b' the target nodes of 'u' in graph G_b).
    
    The merge is a union of the two dictionaries:
    if a target node is only present in 
    one of the two, it will also be present in the
    merged dictionary; 
    if a target node is present in both,
    the weights of the two are summed and it is given
    as timestamp the earlier of the two.
    
    Arguments
        a, b : (dict)
    
    Returns
        (dict) union of a and b
    """
    
    if not a:
        return b
    
    if not b:
        return a
    
    c = a.copy()
        
    # for every key in the 'b' dictionary
    # checks if that key was already present in 'a'
    # and if the value of its time is lower 
    # than what was already present, and in that case replaces it
    # and sums the weights of the two;
    # if not already present then adds it
    for k in b.keys():    
        try:         
            if b[k]["time"] < c[k]["time"]:
                c[k]["time"] = b[k]["time"]
                c[k]["weight"] = a[k]["weight"] + b[k]["weight"]
                
        except KeyError:            
            c[k] = {}
            c[k]["time"] = b[k]["time"]
            c[k]["weight"] = b[k]["weight"]

    return c
    

def merge_graphs(G, J):
    """
    Merges the nodes and edges of two graphs G and J;
    the weight of each edge in the final graph
    is also updated depending on whether the edge 
    was in both graph or just in one
    
    Arguments
        G, J : graphs
    
    Returns
        merged graph
    """
    
    new_graph = Graph()
    
    # all unique nodes in both graphs
    all_nodes = set(list(G.nodes.keys()) + list(J.nodes.keys()))
    
    for src in all_nodes:
        
        # find all edges in both graphs for that node
        try:
            G_node_edges = G[src]
        except KeyError:
            G_node_edges = {}
            
        try:
            J_node_edges = J[src]
        except KeyError:
            J_node_edges = {}
            
        # ...then merge them to get a unique dictionary 
        node_edges = merge_edges(G_node_edges, J_node_edges)
           
        # add those edges to the new graph
        for tgt in node_edges:
            new_graph.add_edge(src, tgt)
            try:
                new_graph[src][tgt]["time"] = node_edges[tgt]["time"]
                new_graph[src][tgt]["weight"] = node_edges[tgt]["weight"]
            except KeyError:
                pass
            
    return new_graph    
        

def plot_neighbors(G, central_node, 
                   neighbors=[], max_neighbors=12, figsize=(8, 8)):
    """
    Plots a subset of the neighbors of a given node
    with the corresponding edges
    
    Arguments
        G             : graph
        central_node  : (str) a node
        neighbors     : (list of str) subset of neighbors of 'central_node'
        max_neighbors : if 'neighbors' is not provided,
                        max number of random neighbors of 'central_node'
                        to be plotted
        figsize       : (tup) figsize of the plot
    Returns
        matplotlib.pyplot object
    """

    if neighbors:
        neighbors = list(map(str, neighbors))
    else:
        neighbors = random.sample(list(G.neighbors[central_node]), 
                                  max_neighbors)

    # nodes for which 'central_node' is the source
    target_nodes = [node for node in neighbors
                    if node in G[central_node].keys()]
    
    # nodes for which 'central_node' is the target
    source_nodes = [node for node in neighbors
                    if node not in target_nodes]

    # angle such that the neighboring nodes
    # will be equidistant around the circle
    # with center the 'central_node'
    angle = 2 * np.pi / len(neighbors)
    
    # coordinates [0,1] and label of those points
    points = [(np.cos(idx * angle), np.sin(idx * angle), label) 
              for idx, label in enumerate(neighbors)]
    
    points.append((0, 0, central_node))
    
    plt.figure(figsize = figsize)
    plt.title(f"Neighbors of {central_node}\n")
    plt.rc('axes', titlesize=20) 
    plt.rc('font', size=15)
    
    plt.xlim((-1.2,1.2))
    plt.ylim((-1.2,1.2))
    
    # arrows
    for x, y, label in points[:-1]:
        
        #segment
        plt.plot([x, 0] ,[y, 0], 
                 color="black",
                 linewidth=1.2)
        
        # True if the arrow has to go from the neighbor
        # to the central_node, False if opposite direction
        to_center = True if label in source_nodes else False
        
        # arrows
        if to_center:
            plt.arrow(x, y, x/2-x, y/2-y,
                      width=0.024,                    
                      length_includes_head = False,
                      edgecolor="white", facecolor="black")
        else:
            plt.arrow(0, 0, x-x/2, y-y/2,
                      width=0.024,
                      length_includes_head = False,
                      edgecolor="white", facecolor="black")   
    
    # nodes and labels
    for x, y, label in points:
        plt.plot(x, y, marker="o", markersize=60)
        plt.text(x, y, label)         
        
    plt.axis('off')
    
    return plt
