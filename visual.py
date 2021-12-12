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