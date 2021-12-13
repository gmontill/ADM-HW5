import pandas as pd
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
        
        metric = input("Metric [btw | pagerank | cc | dc]")
        
        return best_users(G, [time_start, time_end], metric)
        
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
        print("Start and Goal are the same Node!")
        return
    while queue:
        path = queue.pop(0)
        node = path[-1]
        
        if node not in seen:
            neighbours = G[node]
            for neighbour in neighbours:
                new = list(path)
                new.append(neighbour)
                queue.append(new)
                if neighbour == goal:
                    print(*new)
                    return new, len(new)
            seen.append(node)
    print("A connecting path does not exists")
    return
 
def beetweenness():
   
def closeness(G, u):
    summ = 0
    for v in G.nodes:
        path, distance = shortest_path(G, u, v)
        if(distance != 0):
            summ += distance
    if(summ > 0): 
        return((len(G.nodes)-1)/summ)
    else:
        print("Denominator = 0")
        return

def best_users(G, time_interval, metric, v):
    '''
    Returns the value of the given 
    metric applied over the complete 
    graph for the given interval of
    time
    '''
    summ_v = 0
    summ = 0
    
    for p1 in G.nodes:
        for p2 in G.nodes:
            if(pd.to_datetime(G.nodes[p1][p2]['time'], format='%Y-%m-%d') < time_interval[0]
               and pd.to_datetime(G.nodes[p1][p2]['time'], format='%Y-%m-%d') > time_interval[1]):
                break
            path, distance = shortest_path(G, p1, p2)
            if(distance != 0 or distance != float('inf')):
                summ += distance
                if(v in path):
                    summ_v += distance
                
            