import pandas as pd
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    """
    Implementation of a graph
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
        
        # whether the graph is directed or undirected
        # (to double-check and update use the is_directed function)
        self.directed = True
        
        return
    
    def __len__(self):
        """
        Returns the number of nodes
        in the graph
        """
        
        return len(self.nodes.keys())
    
    def __getitem__(self, label):
        """
        Returns the node with label 'label',
        i.e. the dictionary containing all of
        its target nodes and timestamps and weights
        of each of those edges
        
        Arguments
            label : (int), (str) or whatever other object
            
        Returns
            (dict)
        """
        
        return self.nodes[str(label)]
    
    def summary(self):
        """
        Prints a summary of the graph 
        with the total number of nodes and edges
        """
        
        if self.directed:
            print(f"Directed graph with {len(self.nodes)} nodes and {len(self.edges)} edges")
        else:
            # since (u,v) and (v,u) constitute one edge
            # the length of 'self.edges' has to be halved
            print(f"Undirected graph with {len(self.nodes)} nodes and {0.5 * len(self.edges):.0f} edges")
            
    
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
        
        # if the graph is directed
        # and there is already an edge (tgt -> src)
        # then don't add this one (src –> tgt)
        try:
            if self.directed and self.nodes[tgt][src]:
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
    
    def degree(self, node):
        return len(self.neighbors[node])

    
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


def graph_from_df(df, kind="custom", graph_name=""):
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
        G.name = graph_name
    
    # improve 
    for idx in df.index: 
        row = df.iloc[idx]
        
        src = str(row["src"])
        tgt = str(row["tgt"])
        
        # ignore loops
        if src != tgt:           
            G.add_edge(src, tgt)         
            try:
                G[src][tgt]["time"] = row["timestamp"]
                G[src][tgt]["weight"] = 1
            except KeyError:
                pass
 
    return G


def is_directed(G):
    """
    Checks whether a graph is directed
    or undirected
    
    Arguments
        G : graph
    
    Returns
        (bool)
    """
    
    for u, v in G.edges:
        if (v, u) not in G.edges:
            return True
        
    return False


def is_dense(G):
    """
    Checks whether a graph is dense (|E| ~ |V|^2)
    or sparse (|E| << |V|^2)
    
    Arguments
        G : graph
        
    Returns
        (bool)
    """
    
    return len(G.edges) >= 0.5 * len(G.nodes)


def filter_graph_by_time(G, time_interval):
    """
    Filters a graph by only considering
    edges within the provided time interval
    
    Arguments
        G : graph
        time_interval : list-like with dates as Timestamp objects
    
    Returns
        filtered graph
    """
    
    Gf = Graph()
    
    time_start = time_interval[0]
    time_end = time_interval[1]
    
    for (u, v) in G.edges:   
        if G[u][v]["time"] > time_start and G[u][v]["time"] < time_end:
            Gf.add_edge(u, v)
            Gf[u][v] = G[u][v]
            
    return Gf


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
        G, J      : graphs
    
    Returns
        new_graph : merged graph G U J
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


def undirected_graph(G):
    """
    Turns a directed graph G into 
    an undirected one
    
    Arguments
        G  : directed graph
        
    Returns
        Gu : undirected version of G
    """
    
    if not G.directed:
        return G
    
    Gu = Graph()
    Gu.directed = False
    
    for node in G.nodes.keys():
        
        for neigh in G.neighbors[node]:
            
            # for each pair (node, neighbor)
            # adds an edge (node –> neighbor)
            # and an edge (neighbor –> node)
            # unless they were already added earlier
            
            try:
                if Gu[node][neigh]:
                    pass        
            except KeyError:       
                Gu.add_edge(node, neigh)    
                
                try:
                    Gu[node][neigh]["time"] = G[node][neigh]["time"]
                    Gu[node][neigh]["weight"] = G[node][neigh]["weight"]  
                except KeyError:
                    Gu[node][neigh]["time"] = G[neigh][node]["time"]
                    Gu[node][neigh]["weight"] = G[neigh][node]["weight"]  
            
            try:
                if Gu[neigh][node]:
                    pass       
            except KeyError:    
                Gu.add_edge(neigh, node) 
                
                try:
                    Gu[neigh][node]["time"] = G[neigh][node]["time"]
                    Gu[neigh][node]["weight"] = G[neigh][node]["weight"] 
                except KeyError:
                    Gu[neigh][node]["time"] = G[node][neigh]["time"]
                    Gu[neigh][node]["weight"] = G[node][neigh]["weight"] 
            
    return Gu
        

def plot_neighbors(G, central_node, 
                   neighbors=[], max_neighbors=12, figsize=(8, 8)):
    """
    Plots a subset of the neighbors of a given node
    with the corresponding edges
    
    Remark: this plot will only show the edges between
            the central node and its neighbors,
            ignoring those between the neighbors only
    
    Arguments
        G             : graph
        central_node  : (str) a node
        neighbors     : (list of str) subset of neighbors of 'central_node'
        max_neighbors : (int) if 'neighbors' is not provided,
                        max number of random neighbors of 'central_node'
                        to be plotted
        figsize       : (tup) figsize of the plot
        
    Returns
        matplotlib.pyplot object
    """
    
    ######### determine which nodes/edges to plot
    ######### and what coords they should have

    if neighbors:
        
        neighbors = list(map(str, neighbors))
        
    else:
        
        if max_neighbors > len(G.neighbors[central_node]):
            max_neighbors = len(G.neighbors[central_node])

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
    
    ######### PLOT
    
    plt.figure(figsize=figsize)
    plt.title(f"Neighbors of {central_node}")
    plt.rc('axes', titlesize=20) 
    plt.rc('font', size=15)
    
    plt.xlim((-1.2,1.2))
    plt.ylim((-1.2,1.2))
    
    # ALSO ADD WEIGHTS TO THE VISUALIZATION?
    
    # arrows/segments
    for x, y, label in points[:-1]:
        
        # segment from the center to (x,y)
        plt.plot([x, 0] ,[y, 0], 
                 color="black",
                 linewidth=1.2)
        
        # if the graph is directed
        # also add arrows
        if G.directed:      
            # True if the arrow has to go from the neighbor
            # to the central_node, False if opposite direction
            to_center = True if label in source_nodes else False
            
            if to_center:
                plt.arrow(x, y, -x/2, -y/2,
                          width=0.024,                    
                          length_includes_head=False,
                          edgecolor="white", facecolor="black")
            else:
                plt.arrow(0, 0, x/2, y/2,
                          width=0.024,
                          length_includes_head=False,
                          edgecolor="white", facecolor="black")   
    
    # nodes and labels
    for x, y, label in points:
        plt.plot(x, y, marker="o", markersize=60)
        plt.text(x, y, label)         
        
    plt.axis('off')
    
    return plt


def plot_subgraph(G, nodes=[], filename=""):
    """
    Plots a subgraph of the given graph
    with the given nodes/edges
    
    Arguments
        G        : graph
        nodes    : (list) nodes to be plotted
        filename : (str) output of the plot to ext file
        
     Returns
         matplotlib.pyplot object
    """
    
    if nodes:
        nodes = list(map(str, nodes))  
    else:
        rand_node = random.choice(list(G.nodes.keys()))
        nodes = list(G.neighbors[rand_node])
    
    plt.figure(figsize=(0.8*len(nodes), 0.8*len(nodes)))
    plt.rc('font', size=15)
    
    # random coordinates for each of the nodes
    points = {node: tuple(map(float, np.random.normal((2,1))))
                     for node in nodes}
    
    # edges between the given nodes
    edges = [edge for edge in G.edges 
             if edge[0] in nodes and edge[1] in nodes]
    
    # segments
    for node in points.keys():   
        for edge in edges:
            if node in edge:
                x1, y1 = points[edge[0]]
                x2, y2 = points[edge[1]]
                
                plt.plot([x1, x2] ,[y1, y2], 
                         color="black",
                         linewidth=1.2)
                
                # arrows
                if G.directed:
                    plt.arrow(x1, y1, (x2-x1)*0.5, (y2-y1)*0.5,
                              width=0.020,                    
                              length_includes_head=False,
                              edgecolor="white",                 
                              facecolor="black")
     
    # nodes
    for node, (x,y) in points.items():   
        plt.plot(x, y, marker="o", markersize=20)
        plt.text(x, y, node)
            
    plt.axis('off')
    
    if filename:
        plt.savefig(filename)
    
    return plt
