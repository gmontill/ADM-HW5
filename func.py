import pandas as pd
import graph


def func_selector(df, func_num):
    """
    Interface to select which
    functionality of this module use
    
    Arguments
        G        : graph
        func_num : (int) number from 1 to 4
        
    Returns
        output of selected functionality
    """
    filtered_df = df.copy()
    
    if func_num == 1:
        G = graph.graph_from_df(df)
        return overall_features(G)
        
    elif func_num == 2:
        
        node = input("Node")
        
        # add try/except
        time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]"), format='%Y-%m-%d')
        time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]"), format='%Y-%m-%d')
        
        
        filtered_df = filtered_df[(filtered_df["time"] > time_start) & (filtered_df["time"] < time_end)]
        
        G = graph.graph_from_df(filtered_df)
        
        metric = input("Metric [btw | pagerank | cc | dc]")
        
        return best_users(G, [time_start, time_end], metric, node)
        
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
                    return new
            seen.append(node)
    print("A connecting path does not exists")
    return
 
def beetweenness(G, v):
    summ_v = 0
    summ = 0
    count = 0
    
    for p1 in G.nodes:
        for p2 in G.nodes:
            path = shortest_path(G, p1, p2)
            if(len(path) != float('inf') or len(path) != 0):
                summ += 1
                if(v in path):
                    summ_v += count
            else:
                print("Not possible to compute the betweeness, an error occurred: denominator is zero!")
                return
    return summ_v/summ

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
    return G.degree(v)/len(G.nodes)

def best_users(G, time_interval, metric, node):
    '''
    Returns the value of the given 
    metric applied over the complete 
    graph for the given interval of
    time
    '''
    if(metric == 'btw'): return beetweenness(G, time_interval, node)
    elif(metric == 'cc'): return closeness(G,time_interval, node)
    elif(metric == 'pagerank'): return pagerank()
    elif(metric == 'dc'):return degree_cent()
        
    
                
            