import pandas as pd
import numpy as np
from scipy.linalg import eig
import graph


def func_selector(G, func_num):
    """
    Interface to select which
    functionality of this module use
    
    Arguments
        G        : graph
        func_num : (int) number from 1 to 4
        
    Returns
        output of selected functionality
    """
    if func_num == 1:
        return overall_features(G)
        
    elif func_num == 2:
        
        node = input("Node")
        
        # add try/except
        time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]"), format='%Y-%m-%d')
        time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]"), format='%Y-%m-%d')
        
        Gf = graph.filter_graph_by_time(G, [time_start, time_end])
        
        metric = input("Metric [btw | pagerank | cc | dc]")
        
        return best_users(Gf, metric, node)
        
    elif func_num == 3:
        pass
    elif func_num == 4:
        pass

    return
    

def overall_features(G):
    """
    Creates a table with some of 
    the features of a given graph
    
    Arguments
        G : graph
        
    Returns
        (tuple) pandas dataframe and list with the degrees
    """
    
    answers = [G.directed,
               len(G.nodes),
               len(G.edges) if G.directed else int(0.5 * len(G.edges)),
               f"{len(G.edges) / len(G.nodes):.2f}",
               graph.is_dense(G)]
    
    table = pd.DataFrame({"Graph features": answers})

    table.index = ["Directed", 
                   "Users", 
                   "Answers/comments", 
                   "Avg links per user", 
                   "Dense (|E| ~ |V|^2)"]
    
    density = [G.degree(node) for node in G.nodes.keys()]
        
    return table, density
   
    
def shortest_path(G, start, goal):
    seen = []
    queue = [[start]]
    if start == goal:
        return
    while queue:
        path = queue.pop(0)
        node = path[-1]
        
        ########
        # neighbors non cambiano tra dir e indir
        # mentre gli edges sì
        # quale va usato???
        # se va usato edges meglio rinominare
        # qua sotto "neighbor" con "target"
        ########
        if node not in seen:
            neighbours = G[node]
            for neighbour in neighbours:
                #print(neighbour)
                new = list(path)
                new.append(neighbour)
                queue.append(new)
                if neighbour == goal:
                    return new
            seen.append(node)
    return
 
    
def beetweenness(G, v):
    summ_v = 0
    summ = 0
    count = 0
    
    for p1 in G.nodes:
        for p2 in G.nodes:
            path = shortest_path(G, p1, p2)
            if(path and (len(path) != float('inf') or len(path) != 0)):
                print(len(path))
                print(path)
                summ += 1
                if(v in path):
                    summ_v += count
            else:
                break
    if(summ != 0):
        return summ_v/summ
    else:
        print('errore, denominatore = 0')

        
def closeness(G, u):
    summ = 0
    for v in G.nodes:
        path, distance = shortest_path(G, u, v)
        if(distance != 0):
            summ += distance
    if(summ != 0): 
        return((len(G.nodes)-1)/summ)
    else:
        print("Denominator = 0")
        return


def degree_centrality(G, v):
    """
    Degree centrality of a node,
    defined as the defined of the node
    normalized by the max possible degree
    
    Arguments
        G : graph
        v : (str) a node
    
    Returns
        (float) degree centrality of v in G
    """

    return G.degree(v) / (len(G) - 1)


def index_of_node(G, node):
    """
    Returns the index of the given node
    in the adjagency matrix
    
    Arguments
        G    : graph
        node : (str) a node
        
    Returns
        (int) index of the node
    """
    
    return np.where(np.array(list(G.nodes.keys()))==node)[0][0]


def pagerank(G, node):
    """
    Computes the rank of a node
    with a PageRank algorithm
    by finding the eigenvalues and
    eigenvectors of the adjacency matrix;
    the rank will be the component of the
    principal eigenvector corresponding
    to the given node
    
    Arguments
        G    : graph
        node : (str) a node
    
    Returns
        (float) rank of the node
    """
    
    # initialize matrix 
    stoc_mat = np.zeros([len(G), len(G)])
    
    # fill with uniform probability
    for idx, u in enumerate(G.nodes.keys()):
        for v in G[u]:
            stoc_mat[idx][index_of_node(G, v)] = 1/len(G[u])
            
    eigenval, left_eigenvec = eig(stoc_mat, left=True, right=False)
    
    # index in eigenval of the largest eigenvalue
    max_eigenval_idx = np.argmax(eigenval)
    
    # eigenvector for that eigenvalue
    # i.e. the principal eigenvector
    princ_eigenvec = left_eigenvec[:,max_eigenval_idx]
    
    # normalize so it sums up to 1
    princ_eigenvec = princ_eigenvec / sum(princ_eigenvec)
    
    # component in left_eigenvec corresponding to that node
    prob = np.real(princ_eigenvec[index_of_node(G, "238")])
    
    return prob


def best_users(G, metric, node):
    """
    Returns the value of the given 
    metric applied over the complete 
    graph for the given interval of
    time
    
    Arguments
        G      : graph
        metric : (str) btw/cc/pagerank/dc
        node   : (str) a node
        
    Returns
        (float) output of the selected metric
    """

    if (metric == 'btw'): 
        return beetweenness(G, node)
    elif (metric == 'cc'): 
        return closeness(G, node)
    elif (metric == 'pagerank'): 
        return pagerank(G, node)
    elif (metric == 'dc'):
        return degree_centrality(G, node)
        
    
                
            