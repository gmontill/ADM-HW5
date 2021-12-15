import pandas as pd
import graph

def filter_graph_by_time(G, time_interval):
    
    Gf = graph.Graph()
    
    time_start = time_interval[0]
    time_end = time_interval[1]
    
    for (u,v) in G.edges:
        if G[u][v]["time"] > time_start and G[u][v]["time"] < time_end:
            Gf.add_edge(u, v)
            Gf[u][v] = G[u][v]
            
    return Gf

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
        
        time_int = (time_start, time_end)
        
        Gf = filter_graph_by_time(G, time_int)
        
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
        
        if node not in seen:
            neighbours = G[node]
            for neighbour in neighbours:
                print(neighbour)
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

def deg_cent(G, v):
    s = 1.0 / (len(G) - 1)
    return s * G.degree(v)

def best_users(G, metric, node):
    '''
    Returns the value of the given 
    metric applied over the complete 
    graph for the given interval of
    time
    '''
    if(metric == 'btw'): return beetweenness(G, node)
    elif(metric == 'cc'): return closeness(G, node)
    elif(metric == 'pagerank'): return pagerank()
    elif(metric == 'dc'):return deg_cent(G, node)
        
    
                
            