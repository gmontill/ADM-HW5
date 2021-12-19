import pandas as pd
import graph
import func
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

# Functions to visualize the outputs
# of the functionalities in `func.py`

def overall_features(G, table, density):
    """
    Displays a table with some of 
    the features of a given graph
    and plots a histogram of the degrees
    
    Arguments
        G       : graph
        table   : pandas dataframe
        density : (list) degrees of all nodes of G
        
    Returns
        matplotlib.pyplot object
    """
    
    display(table)
    
    plt.title("Degree distribution")
    plt.xlabel("Degree")
    sns.histplot(density, stat="probability")
    
    return plt


def metric_evolution(G, node, metrics):
    """
    Computes and then plots the evolution
    of the given metrics for a certain node
    in a graph
    
    Arguments
        G : a graph
        node : a node
        metrics : (list) of metrics [dc|cc|btw|pagerank]
        
    Returns
        matplotlib.pyplot object
    """
    
    time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]"), format='%Y-%m-%d')
    time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]"), format='%Y-%m-%d')
    
    inters = list(pd.date_range(time_start, time_end, freq = "M"))

    if time_start != inters[0]:
        inters.insert(0, time_start)
    if time_end != inters[-1]:
        inters.append(time_end)
    
    inters[0] -= pd.Timedelta('1d')
    inters = zip(map(lambda d: d + pd.Timedelta('1d'), inters[:-1]), inters[1:])
    intervals = list(map(lambda t: t[0].strftime('%Y-%m-%d') + ' - ' + t[1].strftime('%Y-%m-%d'), inters))
    
    intervals_datetime = []
    for interval in intervals:
        int_start, int_end = interval.split(' - ')
        intervals_datetime.append([pd.to_datetime(int_start, format='%Y-%m-%d'),
                                   pd.to_datetime(int_end, format='%Y-%m-%d')])
    
    plt.figure(figsize = (10,6))
    
    plt.xlabel("Time intervals")
    plt.title(f"Centrality of {node} over time")
    plt.ylabel("Metric")
    plt.xticks(ticks=range(len(intervals)), 
               labels=intervals, 
               rotation=45)
        
    for metric in metrics:
        values = []
        for interval in intervals_datetime:
            Gm = graph.filter_graph_by_time(G, interval)        
            try:
                if Gm[node]:
                    values.append(func.best_users(Gm, metric, node))
            except KeyError:
                pass
        plt.plot(range(len(intervals)), values)
        plt.scatter(range(len(intervals)), values)

    plt.legend(metrics)
    
    return plt
