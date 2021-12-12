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
    
    
def best_users(G, time_interval, metric):
    pass