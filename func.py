import pandas as pd
import numpy as np
from queue import deque
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
        
        node = input("Node: ")
        
        # add try/except
        time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]: "), format='%Y-%m-%d')
        time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]: "), format='%Y-%m-%d')
        
        Gf = graph.filter_graph_by_time(G, [time_start, time_end])
        
        metric = input("Metric [btw | pagerank | cc | dc]: ")
        
        return best_users(Gf, metric, node)
        
    elif func_num == 3:
        start = input("Start Node: ")
        end = input("End Node: ")
        
        users = input("Nodes to be visited (comma-separated)").split(",")
        
        time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]: "), format='%Y-%m-%d')
        time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]: "), format='%Y-%m-%d')
        
        if (time_start != '' and time_end != ''):
            Gf = graph.filter_graph_by_time(G, [time_start, time_end])
            ord_route(Gf, users, start, end)
            
    elif func_num == 4:
        time_start1 = pd.to_datetime(input("Time start for the 1st interval [yyyy-mm-dd]: "), format='%Y-%m-%d')
        time_end1 = pd.to_datetime(input("Time end for the 1st interval [yyyy-mm-dd]: "), format='%Y-%m-%d')

        time_start2 = pd.to_datetime(input("Time start for the 2nd interval [yyyy-mm-dd]: "), format='%Y-%m-%d')
        time_end2 = pd.to_datetime(input("Time end for the 2nd interval [yyyy-mm-dd]: "), format='%Y-%m-%d')

        Gf1 = graph.filter_graph_by_time(G, [time_start1, time_end1])
        Gf2 = graph.filter_graph_by_time(G, [time_start2, time_end2])

        user1 = user_finder(Gf1, Gf2)
        user2  = user_finder(Gf2, Gf1)

        Gf = graph.merge_graphs(Gf1, Gf2)

        return disc_graph(Gf, user1, user2)

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


# dijkstra 
def shortest_path(G, start, goal):
    
    start = str(start)
    goal = str(goal)
    
    if start == goal:
        return [start, goal]
    
    visited = {str(node):False for node in G.nodes.keys()}
    dist = {str(node):len(G.edges) for node in G.nodes.keys()}
    prev = {str(node):None for node in G.nodes.keys()}
    path = []
    
    visited[start] = True
    dist[start] = 0
    
    Q = deque()
    Q.append(start)
    
    while Q:
        
        node = Q.popleft()    
        
        if node == goal:
            v = prev[goal]
            
            while v != start:
                path.insert(1, v)
                v = prev[v]  
                
            path.insert(0, start)
            path.append(goal)
            
            return path
        
        for target in G[node].keys():
            
            if not visited[target]:
                visited[target] = True   
                
                if dist[target] > dist[node] + G[node][target]["weight"]:
                    
                    dist[target] = dist[node] + G[node][target]["weight"]
                    prev[target] = node
                    
                    Q.append(target)
    
    return []


#https://www.baeldung.com/cs/graph-number-of-shortest-paths
def num_of_shortest_paths(G, start, goal, intermed=None):
    
    dist = {str(node):len(G.edges) for node in G.nodes.keys()}
    paths = {str(node):0 for node in G.nodes.keys()}
    
    priority_q = deque()
    priority_q.append((0, start))
    
    dist[start] = 0
    paths[start] = 1
    
    while priority_q:
        length, current = priority_q.pop()
        
        for target in G[current].keys():
            weight = G[current][target]["weight"]
            
            if dist[target] > dist[current] + weight:
                priority_q.append((length, target))
                dist[target] = dist[current] + weight
                paths[target] = paths[current]

            elif dist[target] == dist[current] + weight:
                paths[target] += paths[current]

    if intermed:
        return paths[goal], paths[intermed]
    else:
        return paths[goal]
        

def betweenness(G, v):
    
    btw = 0
    
    for s in G.nodes.keys():
        for t in G.nodes.keys():
            if s != t and t != v and s != v:
                num_sp = num_of_shortest_paths(G, start=s, goal=t, intermed=v)
                if num_sp[0]:
                    btw += num_sp[1] / num_sp[0]                 
                
    return btw
            

        
def closeness(G, u):
    summ = 0
    for v in G.nodes:
        path = shortest_path(G, u, v)
        if(path):
            summ += len(path)
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
    
    # component in princ_eigenvec corresponding to that node
    prob = np.real(princ_eigenvec[index_of_node(G, node)])
    
    return prob


def ord_route(G, users, start, end):
    path_ste = shortest_path(G, start, end)
    if not path_ste:
        print("Not possible to find the ordered route because start and stop can't be connected!")
        return
    
    out_path= []
    out_path.append(start)
    users.insert(0, start)
    users.append(end)
    
    for u in range(len(users) - 1):
        path = shortest_path(G, users[u], users[u+1])
        if not path:
            print(f"Not possible to find the ordered route because {users[u]} and {users[u+1]} can't be connected")
        out_path += path[1:]
    
    print(out_path)
    
    return out_path


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
        return betweenness(G, node)
    elif (metric == 'cc'): 
        return closeness(G, node)
    elif (metric == 'pagerank'): 
        return pagerank(G, node)
    elif (metric == 'dc'):
        return degree_centrality(G, node)


def user_finder(G1, G2):
    '''
    Parameters
    ----------
    G1 : Graph 1
    G2 : Graph 2
    Returns
    -------
    A node (user) which is in G1 but not in G2
    '''
    for user in G1.nodes:
        if user not in G2.nodes:
            return user
    return f"Couldn't find a node present only in {G1}"


def disc_graph(G, user1, user2):
    path = shortest_path(G, user1, user2)

    if path == []:
        return "The nodes are already disconnected"
    w = 0
    out = 0 
    edge = [path[0], path[1]]
    while path != []:
        for n in range(len(path) - 1):
            # aggiungere controllo sui pesi
            edge = [path[n], path[n+1]]
        out += 1
        G.remove_edge(*edge)
        path = shortest_path(G, user1, user2)

    return f"the minimum number of links is {out}" 
