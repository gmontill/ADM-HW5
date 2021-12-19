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


def draw_short_ord_route(result):

    '''
    result: result list of the funct 3 
    
    output a graph that connects 
    '''

    plt.rcParams["figure.figsize"]=(15,15)
    G = nx.DiGraph()
    for i in range(len(result)-1):
        G.add_edge(result[i],result[i+1], label=i)

    edge_labels = nx.get_edge_attributes(G, 'label')
    #print('edge_labels:', edge_labels)

    if not nx.is_empty(G):
        pos=nx.spring_layout(G)

        nx.draw(G,pos, with_labels=True, font_size=25,connectionstyle='arc3,rad=0.2', arrowsize=25)
        colors = ["red"] + (["green"] * (len(result) - 1))

        nx.draw_networkx_nodes(G,pos, node_size = 650, node_color=colors)


        plt.show()  
        