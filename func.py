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
        
        time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]: "), 
                                    format='%Y-%m-%d')
        time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]: "), 
                                  format='%Y-%m-%d')
        
        Gf = graph.filter_graph_by_time(G, [time_start, time_end])
        
        metric = input("Metric [btw | pagerank | cc | dc]: ")
        
        return best_users(Gf, metric, node)
        
    elif func_num == 3:
        start = input("Start Node: ")
        end = input("End Node: ")
        
        users = input("Nodes to be visited (comma-separated): ").split(",")
        
        time_start = pd.to_datetime(input("Time start [yyyy-mm-dd]: "), 
                                    format='%Y-%m-%d')
        time_end = pd.to_datetime(input("Time end [yyyy-mm-dd]: "), 
                                  format='%Y-%m-%d')
        
        if (time_start != '' and time_end != ''):
            Gf = graph.filter_graph_by_time(G, [time_start, time_end])
            return ord_route(Gf, users, start, end)
            
    elif func_num == 4:
        time_start1 = pd.to_datetime(input("Time start for the 1st interval [yyyy-mm-dd]: "), 
                                     format='%Y-%m-%d')
        time_end1 = pd.to_datetime(input("Time end for the 1st interval [yyyy-mm-dd]: "), 
                                   format='%Y-%m-%d')

        time_start2 = pd.to_datetime(input("Time start for the 2nd interval [yyyy-mm-dd]: "), 
                                     format='%Y-%m-%d')
        time_end2 = pd.to_datetime(input("Time end for the 2nd interval [yyyy-mm-dd]: "), 
                                   format='%Y-%m-%d')

        Gf1 = graph.filter_graph_by_time(G, [time_start1, time_end1])
        Gf2 = graph.filter_graph_by_time(G, [time_start2, time_end2])
        
        user1 = input("User 1: ")
        user2 = input("User 2: ")

        if not user1:
            user1 = user_finder(Gf1, Gf2)
            print(f"\nA user unique to G_1 is {user1}")
        if not user2:
            user2  = user_finder(Gf2, Gf1)
            print(f"A user unique to G_2 is {user2}\n")

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


def shortest_path_uw(G, start, goal):
    """
    Computes the shortest path between
    two nodes in an unweighted graph
    
    Arguments
        G     : graph
        start : starting node
        goal  : target node
        
    Returns
        (list) shortest path
    """
    
    seen = []
    queue = [[start]]
    
    if start == goal:
        print("Start and goal are the same node!")
        return []
    
    while queue:
        path = queue.pop(0)
        node = path[-1]

        if node not in seen:
            neighbours = G[node]
            for target in neighbours:
                new = list(path)
                new.append(target)
                queue.append(new)
                if target == goal:
                    return new
            seen.append(node)
    return []


def shortest_path(G, start, goal):
    """
    Finds the shortest path between
    two nodes of a weighted graph
    using the Dijkstra algorithm
    
    Arguments
        G     : a weighted graph
        start : the starting node
        goal  : the target node
    
    Returns
        (list) list of nodes in the path
    """
    
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


def num_of_shortest_paths(G, start, goal, intermed=None):
    """
    Computes the number of shortest paths
    between two nodes of a weighted graph;
    two paths are considered equally short
    if the total weights of the edges is the same,
    regardless of the actual number of edges.
    
    The following code is based on the pseudocode
    on this page:
    https://www.baeldung.com/cs/graph-number-of-shortest-paths
    
    Arguments
        G        : a weighted graph
        start    : starting node
        goal     : target node
        intermed : if indicated the function also returns
                   the number of paths start -> goal
                   which also include this node
    
    Returns
        (int) num of shortest paths
        or if 'intermed' is indicated
        (tuple) num of shortest paths and
                how many of those include 'intermed'
    """ 
    
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
    """
    Computes the betweenness centrality
    of a node in a weighted graph,
    as defined by Freeman (1977), Anthonisse (1971)
    
    Arguments
        G : a weighted graph
        v : a node
        
    Returns
        (float) normalized betweenness centrality
    """
    
    btw = 0
    
    for s in G.nodes.keys():
        for t in G.nodes.keys():
            if s != t and t != v and s != v:
                num_sp = num_of_shortest_paths(G, start=s, goal=t, intermed=v)
                if num_sp[0]:
                    btw += num_sp[1] / num_sp[0]  
                    
    # normalization
    if G.directed:            
        return btw / ((len(G)-1) * (len(G)-2))
    else:
        return 2 * btw / ((len(G)-1) * (len(G)-2))
        

def closeness(G, u):
    """
    Computes the closeness centrality
    of a node in graph, as defined by
    Sabidussi (1966)
    
    Arguments
        G : a graph
        u : a node
        
    Returns
        (float) normalized closeness centrality
    """
    
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
    defined as the degree of the node
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

def adj_function(G, u, v):
    """
    Computes the ratio between the
    number of edges outbound from node v to node u 
    to the total number of outbound links of node v
    
    Arguments
        u : a node
        v : a node
        
    Returns
        (float)
    """
    try:
        if G[v][u]:
            return 1/len(G[v].keys())
    except KeyError:
        return 0
    

def pagerank(G, node):
    """
    Computes the rank of a node
    with a PageRank algorithm
    by finding the eigenvalues and
    eigenvectors of a modified adjacency matrix;
    the rank will be the component of the
    principal eigenvector corresponding
    to the given node
    
    Arguments
        G    : graph
        node : (str) a node
    
    Returns
        (float) rank of the node
    """
    
    # damping factor
    d = 0.85
    
    # initialize stochastic matrix 
    stoc_mat = np.zeros([len(G), len(G)])
    
    # fill the matrix
    for i, u in enumerate(G.nodes.keys()):
        for j, v in enumerate(G.nodes.keys()):
            stoc_mat[i][j] = adj_function(G, u, v)
    
    M = d*stoc_mat + np.ones([len(G),len(G)])*(1-d)/len(G)

    # eigenvalues and right eigenvec of M
    eigenval, eigenvec = eig(M, right=True)
    
    # index in eigenval of the largest eigenvalue
    max_eigenval_idx = np.argmax(eigenval)
    
    # eigenvector for that eigenvalue
    # i.e. the dominant eigenvector
    dom_eigenvec = eigenvec[:, max_eigenval_idx]
    
    # normalize so it sums up to 1 
    # (it's supposed to be a distribution)
    dom_eigenvec = dom_eigenvec / sum(dom_eigenvec)
    
    # component in 'dom_eigenvec' corresponding to that node
    prob = np.real(dom_eigenvec[index_of_node(G, node)])
    
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
        return betweenness(G, node)
    elif (metric == 'cc'): 
        return closeness(G, node)
    elif (metric == 'pagerank'): 
        return pagerank(G, node)
    elif (metric == 'dc'):
        return degree_centrality(G, node)


def ord_route(G, users, start, end):
    """
    Finds the shortest path between two nodes
    which also visits a list of other nodes in order
    
    Arguments
        G     : a graph
        users : list of nodes to be visited in order
        start : starting node
        end   : target node
        
    Returns
        (list) ordered route 
    """
    
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
    
    return out_path


def user_finder(G1, G2):
    """
    Finds a node (a user) which is only in one
    of the two graphs provided
    
    Arguments
        G1, G2 : two graphs
        
    Returns
        a node in G1 and not in G2
    """
    for user in G1.nodes:
        if user not in G2.nodes:
            return user
    
    print("Couldn't find a node present only in G1")
    return


def disc_graph(G, user1, user2):
    """
    Find the minimum number of edges
    required to disconnect two users
    
    Arguments
        G     : a graph
        user1 : a node
        user2 : a node
        
    Returns
        tuple with:
            (int)  number of links
            (list) edges as tuples
            (list) shortest paths between user1 and user2
    """
    
    path = shortest_path_uw(G, user1, user2)
    if path == []:
        return "The nodes are already disconnected"
    
    paths = []
    out = 0
    edge = (path[0], path[1])
    removed = []
    
    while path != []:
        min_w = float('inf')
        tot_w = 0
        
        for n in range(len(path) - 1):
            if G[str(path[n])][str(path[n+1])]['weight'] <= min_w:
                min_w = G[str(path[n])][str(path[n+1])]['weight']
                edge = (path[n], path[n+1])
        
        paths.append(path)
        out += 1    
        removed.append(edge)
        G.remove_edge(edge[0], edge[1])
        
        path = shortest_path_uw(G, user1, user2)
        
    print(f"The minimum number of links required to disconnect the two graphs is {out}")
    
    return out, removed, paths