# -*- coding: utf-8 -*-
"""
This function computes the attribute-conformity of the network and the overall degree of conformity, 
using the clustering of the final probabilities of agents, as attribute of each agent (node of the network)
 The evaluation employes a hierarchical clustering of the final probabilities
 and attributes the resulting cluster to each node (agent) of the network.
 Then, it uses this attribute to compute the node-conformity score for each agent
 and generate a histogram visualization of the conformity scores distribution.
 The overall degree of Conformity of the network is embedded into the conformity distribution plot.

# Author information
__author__ = ["Verdiana Del Rosso"
"Fabrizio Fornari"]
__email__ = [
    "verdiana.delrosso@unicam.it",
    "fabrizio.fornari@unicam.it"
]
"""

from conformity import attribute_conformity
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from sklearn import cluster
import time
import warnings
from exceptions.custom_exceptions import ValidationError, MetricsError

def probabilities_clustering(probabilities, cluster_mode: str = 'auto', clust_algorithm: str = 'average', 
                             dist_threshold: float = 0.05, bins: int =20): 
    """ 
    Perform the clustering of the final beliefs of agents.
    The clustering can be obtained manually or by a hierachical clustering algorithm
    
    Args:
    - probabilities: agents' final probabilities. It is an array of numbers in [0,1] 
    - cluster_mode: options are 'auto' or 'manual'.  In manual mode the range [] is divided in 
      20 bins and the histogram is computed. In Auto mode the hierarchical clustering is used.
    Other params in auto mode:
        - algorithm = 'ward', 'single', 'average', 'complete'; default is 'average'
            ward is the most effective method for noisy data;
            single is fast, and can perform well on non-globular data, but it performs poorly in the presence of noise.
            complete and average perform well on cleanly separated globular clusters, but have mixed results otherwise.
        - dist_threshold = distance threshold for clustering research. Suggested: 0.05 or 0.01
            If this param is 0.05,the dataset is divided in no more than 20 clusters; 
            if it is 0.01, the datatset is divided in maximum 50 clusters. 
    Other parameters for manual mode:
    - bins = number of bins the [0,1] range of subjective probabilities is divided
    
    Returns:
    - cluster_size: array with the size of each cluster
    - y_pred: cluster number of each node
    - t0: start time for clustering
    - t1: end time for clustering
    - label: names of clusters
    """
    if not isinstance(probabilities, (list, np.ndarray)):
        raise ValidationError("Error in probabilities_clustering() for conformity score: Input probabilities must be a list or numpy array.", field='probabilities')
    if not cluster_mode in ['auto', 'manual']:
        raise ValidationError("Error in probabilities_clustering() for conformity score: Invalid cluster_mode. Must be 'auto' or 'manual'.", field='cluster_mode')
    
    match cluster_mode:
        case 'manual':
            try:
                #manual clustering dividing the [0,1] in 20 bins 
                clusters_size, bin_edges = np.histogram(probabilities, bins=bins, range = [0,1], density=False)
                return clusters_size, np.digitize(probabilities, bin_edges)-1, None, None, bin_edges
            except Exception as e:
                raise MetricsError("Error in manual probabilities_clustering() for conformity score: " + str(e), field='probabilities')
        case 'auto':
            # apply a hierarchical clustering to search for clusters into the probabilities array
            X = probabilities.reshape(-1,1) # Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
           
            # normalize dataset for easier parameter selection
            #X = StandardScaler().fit_transform(X)  # performe worse

            # ============
            # Create cluster objects
            # ============
            """
            ward = cluster.AgglomerativeClustering(
                n_clusters=None, linkage="ward", distance_threshold=dist_threshold
            ) # is the most effective method for noisy data
            complete = cluster.AgglomerativeClustering(
                n_clusters=None, linkage="complete", distance_threshold=dist_threshold
            ) # perform well on cleanly separated globular clusters, but have mixed results otherwise.
            average = cluster.AgglomerativeClustering(
                n_clusters=None, linkage="average", distance_threshold=dist_threshold
            ) # perform well on cleanly separated globular clusters, but have mixed results otherwise.
            single = cluster.AgglomerativeClustering(
                n_clusters=None, linkage="single", distance_threshold=dist_threshold
            ) #  is fast, and can perform well on non-globular data, but it performs poorly in the presence of noise.
            """
            try:
                match clust_algorithm:
                    case 'single':
                        clustering_algorithms = cluster.AgglomerativeClustering(
                            n_clusters=None, linkage="single", distance_threshold=dist_threshold
                            ) #  is fast, and can perform well on non-globular data, but it performs poorly in the presence of noise.
                    case 'average':
                        clustering_algorithms = cluster.AgglomerativeClustering(
                            n_clusters=None, linkage="average", distance_threshold=dist_threshold
                            ) # perform well on cleanly separated globular clusters, but have mixed results otherwise.
                    case 'complete':
                        clustering_algorithms = cluster.AgglomerativeClustering(
                            n_clusters=None, linkage="complete", distance_threshold=dist_threshold
                            ) # perform well on cleanly separated globular clusters, but have mixed results otherwise.
                    case 'ward':
                        clustering_algorithms = cluster.AgglomerativeClustering(
                            n_clusters=None, linkage="ward", distance_threshold=dist_threshold
                            ) # is the most effective method for noisy data

                t0 = time.time()

                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="the number of connected components of the "
                        + "connectivity matrix is [0-9]{1,2}"
                        + " > 1. Completing it to avoid stopping the tree early.",
                        category=UserWarning,
                    )
                    clustering_algorithms.fit(X)

                # predict labels
                t1 = time.time()
                if hasattr(clustering_algorithms, "labels_"):
                    y_pred = clustering_algorithms.labels_.astype(int)
                else:
                    y_pred = clustering_algorithms.predict(X)

                labels, clusters_size = np.unique(y_pred,return_counts=True)
                return clusters_size, y_pred, t0, t1, labels
            
            except Exception as e:
                raise MetricsError("Error in Aglomerative algorithm in probabilities_clustering() for conformity score: " + str(e), field='probabilities')

def effective_number_clusters(clusters_size) -> float: 
    """ 
    Perform the effective number of clusters of final beliefs 
    of agents' network.
       
    
    Parameters:
    - clusters_size: array with the size of each cluster
    
    Returns:
    - N_clusters: effective number of clusters
    NOTE: This is equivalent to the function ncluster in "metrics_functions" in AlmondoModel. 
    The difference is in the clustering method
    """
  
  
    # compute the effective number of clusters
    N_clusters = (sum(clusters_size))**2/(sum(clusters_size**2))
    return N_clusters  # Clusters partecipation ratio



def compute_conformity_scores(G: nx.Graph, y_label, alphas=[1.0]) -> tuple[dict[int, float], float]:
    """
    Compute conformity scores for each node in a graph and the overall conformity degree
    based on clustering results.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph
    y_label: array-like
        label of each node
    alphas : list of float, optional (default=[1.0])
        List of alpha parameters that control the level of interaction between nodes.
        alpha = 1.0 imposes linear decrease w.r.t. distance
        alpha > 1.0 imposes sublinear decrease, reducing interaction between distant nodes
    
    Returns:
    --------
    node_conformity: Dictionary mapping nodes to their conformity scores
    global_conformity: Float representing the overall conformity degree

    Notes:
    ------
    The conformity score measures how similar a node is to its neighbors in terms of 
    cluster assignments. A higher score indicates greater conformity with neighboring nodes.
    """
    if len(y_label) != G.number_of_nodes():
        raise ValidationError("Error in compute_conformity_scores(): Length of y_label must match number of nodes in the graph.", field='y_label')

    # Assign cluster labels to nodes
    for i, node in enumerate(G.nodes):
        G.nodes[node]['cluster'] = str(y_label[i])
    
    try:
        # Compute node-level conformity scores
        node_to_conformity = attribute_conformity(G, alphas, ['cluster'], profile_size=1)
        
        # Extract conformity scores for the given alpha (default 1.0)
        node_conformity = node_to_conformity[str(alphas[0])]['cluster']
        
        # Compute global conformity (mean of node conformities)
        global_conformity = sum(node_conformity.values()) / len(node_conformity)
        
        return node_conformity, global_conformity
    except Exception as e:
        raise MetricsError("Error in compute_conformity_scores(): " + str(e), field='G or y_label')

def compute_conformity_scores_opinion(G: nx.Graph, probabilities, alphas=[1.0]) -> tuple[dict[int, float], float]:
    """
    Compute conformity scores for each node in a graph with respect to its opinion (optimistic, pessimistic or neutral)
    and the overall conformity degree based on clustering results.
    
    Parameters:
    -----------
    G : networkx.Graph
        The input graph
    probabilities : array-like
        Probability of agents
    alphas : list of float, optional (default=[1.0])
        List of alpha parameters that control the level of interaction between nodes.
        alpha = 1.0 imposes linear decrease w.r.t. distance
        alpha > 1.0 imposes sublinear decrease, reducing interaction between distant nodes
    
    Returns:
    --------
    node_conformity: Dictionary mapping nodes to their conformity scores
    global_conformity: Float representing the overall conformity degree

    Notes:
    ------
    The conformity score measures how similar a node is to its neighbors in terms of 
    cluster assignments. A higher score indicates greater conformity with neighboring nodes.
    """

    if len(probabilities) != G.number_of_nodes():
        raise ValidationError("Error in compute_conformity_scores_opinion(): Length of probabilities must match number of nodes in the graph.", field='probabilities')

    # Assign cluster labels to nodes
    for i, node in enumerate(G.nodes):
        if probabilities[i] <= 0.33:
            label = 'optimistic'
        elif probabilities[i] > 0.66:
            label = 'pessimistic'
        else:
            label = 'neutral'
        G.nodes[node]['label'] = label
    
    try:
        # Compute node-level conformity scores
        node_to_conformity = attribute_conformity(G, alphas, ['label'], profile_size=1)
        
        # Extract conformity scores for the given alpha (default 1.0)
        node_conformity = node_to_conformity[str(alphas[0])]['label']
        
        # Compute global conformity (mean of node conformities)
        global_conformity = sum(node_conformity.values()) / len(node_conformity)
        
        return node_conformity, global_conformity
    
    except Exception as e:
        raise MetricsError("Error in compute_conformity_scores_opinion(): " + str(e), field='G or y_label')

def plot_conformity_distribution(node_conformity: dict, global_conformity: float,
                                 filename: str = None, ax = None, 
                                 stat: bool = True, title: bool = True,
                                 color: str = 'lightblue', label: str = None, legend: bool = False,
                                 figure_size=(10, 6), grid: bool = False,
                                 transparent_bg: bool = False, transparent_plot_area: bool = False):
    """
    Create a histogram visualization of the conformity scores distribution.
    
    Parameters:
    -----------
    - node_conformity: Dict mapping nodes to their conformity scores
    - global_conformity: Float representing overall conformity degree
    - filename: If provided, the plot will be saved to this file.
    - ax: If provided, the plot will be drawn on this axes.
    - stat: If True, it shows the global conformity on the legend.
    - title: If True, it adds a title to the plot.
    - color: color of barplot and KDE line
    - label: label of plot in legend.
    - legend: If True, it adds the legend to the plot.
    - figure_size: The size of the figure as (width, height) in inches. Default=(10, 6)
    - grid: If True, it adds the horizontal grid to the plot.
    - transparent_bg: If True, the background of the figure will be transparent.
    - transparent_plot_area: If True, the plot area will have a transparent background.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    matplotlib.axes.Axes
        The axes object containing the histogram
    
    Notes:
    ------
    The plot shows:
    - A histogram of node conformity scores as percentages
    - The global degree of conformity as a text annotation, if required
    - X-axis limited to [-1.0, 1.0] for standardized comparison
    """
    # Create figure and axes
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    else:
        fig = ax.get_figure()
    
    data = list(node_conformity.values())
    """
    Add the following lines to adjust the histogram bin width 
    if data has just one unique value
    # Check if all values are the same (or nearly the same) to properly adjust bin width
    if np.std(data) < 1e-6:  # Very small standard deviation means essentially constant
        # For constant data, create custom bins with desired width to avoid a single line
        unique_val = data[0]
        bin_width = 0.015  # Adjust this to control bar width
        bins = [unique_val - bin_width/2, unique_val + bin_width/2]
    else:
        # For distributed data, use regular binning
        bins = 50
    """
    
    label_plot = f"{label} (Conformity: {global_conformity:.2f})" if stat else label
    stat_plot = False # if legend else stat
    # Create histogram using seaborn
    ax = sns.histplot(
        data=data, bins='auto',
        color=color, edgecolor = 'w', alpha=0.6,  # alternative colors: lightgreen, lightsalmon
        stat='percent', ax=ax,  # alternative: stat='frequency'
        kde=True, # compute the kernel density estimate to smooth the distribution and show on the plot as (one or more) line(s)
        line_kws={'color': color,'linestyle':'-','linewidth':2, 'alpha': 1.0}, # 'label': 'KDE'},
        label = label_plot  # dict with Parameters that control the KDE visualization
    )
    
    # eventualmente aggiungere un secondo asse con l'altro plot (per gli agenti pessimisti)
    # Set figure and plot area background
    if transparent_bg: # figure background
        fig.patch.set_facecolor('none')
    else:
        fig.patch.set_facecolor('white')

    if transparent_plot_area: # plot area background
        ax.set_facecolor('none')
    else:
        ax.set_facecolor('white')  # default value

    # Set labels and title
    ax.set_xlabel('Node Conformity', fontsize=12)
    ax.set_ylabel('% Agents', fontsize=12)

    if title:
        ax.set_title('Conformity Score Distribution',fontsize=14)
    
    if grid:
            ax.grid(axis='y')
        
    if stat_plot:
    # Add global conformity annotation
        ax.text(
            0.99,
            0.95,
            f"Global Degree of Conformity {label}: {global_conformity:.2f}",
            transform=ax.transAxes,
            size=12,
            horizontalalignment="right"
        )
    
    # Set x-axis limits
    ax.set_xlim(-1.0, 1.0)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)

    if legend:
        title_legend = "Groups & Conformity Values" if stat else None
        ax.legend(title=title_legend)
    
    # Adjust layout
    plt.tight_layout()
    if filename is not None:
        bg_color = 'none' if transparent_bg else 'white'
        plt.savefig(filename, dpi=300, facecolor=bg_color, bbox_inches='tight')
        fig = plt.gcf()
        ax = plt.gca()
        plt.close()
        return fig, ax
    else:
        return plt.gcf(), plt.gca()
    # plt.close()

def create_conformity_distribution_subplots(sub1_configs, sub3_configs, 
                                  filename: str = None, ax = None, 
                                  stat: bool = True, title: bool = True,
                                  legend: bool = False,
                                  figure_size=(12, 8), grid: bool = False,
                                  transparent_bg: bool = False, transparent_plot_area: bool = False):
    """
    Create a 2x1 subplot with multiple histograms in top plot and single/multiple histograms in bottom
    
    Parameters:
    - sub1_config: list of dictonaries with config data for each histogram in the top subplot
        [{'data': node_conf_data, 'conformity': global:conformity_score, 'color': 'green', 'label': 'Optimistic'}
    - sub2_config: list of dictonaries with config data for each histogram in the bottom subplot (as sub1_config)
    - stat: If True, it shows the global degree of conformity on the legend.
    - title: If True, it adds a title to the plot.
    - color: color of barplot and KDE line
    - label: label of plot in legend.
    - legend: If True, it adds the legend to both the plots.
    - figure_size: The size of the figure as (width, height) in inches. Default=(10, 8)
    - grid: If True, it adds the horizontal grid to the plot.
    - transparent_bg: If True, the background of the figure will be transparent.
    - transparent_plot_area: If True, the plot area will have a transparent background.
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object containing the plot
    matplotlib.axes.Axes
        The axes object containing the subplots
    
    Notes:
    ------
    The plot shows:
    - three overlapped histograms of node conformity scores for optimistist, pessimist and neutral agents as percentages 
        in the top plot
    - A histigram of node conformity score for overall agent probabilities
    - The global degree of conformity as a text annotation in the legend, if required
    - X-axis limited to [-1.0, 1.0] for standardized comparison
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figure_size)
    
    # Set figure and plot area background
    if transparent_bg: # figure background
        fig.patch.set_facecolor('none')
    else:
        fig.patch.set_facecolor('white')

    # Top 1 subplot - Optimistic, pessimistic, neutral histogram
    # for data, color, label in zip(q1_datasets, q1_colors, q1_labels):
    for config in sub1_configs:
        label = config['label']
        global_conformity = config['conformity']
        color = config['color']
        data = config['data']
        label_plot = f"{label} (Conformity: {global_conformity:.2f})" if stat else label
        stat_plot = False # if legend else stat
        # Create histogram using seaborn
        sns.histplot(
            data=data, bins='auto',
            color=color, edgecolor = 'w', alpha=0.6,  # alternative colors: lightgreen, lightsalmon
            stat='percent', ax=ax1,  # alternative: stat='frequency'
            kde=True, # compute the kernel density estimate to smooth the distribution and show on the plot as (one or more) line(s)
            line_kws={'color': color,'linestyle':'-','linewidth':2, 'alpha': 1.0}, # 'label': 'KDE'},
            label = label_plot  # dict with Parameters that control the KDE visualization
        )
        
        # eventualmente aggiungere un secondo asse con l'altro plot (per gli agenti pessimisti)

        if transparent_plot_area: # plot area background
            ax1.set_facecolor('none')
        else:
            ax1.set_facecolor('white')  # default value

        # Set labels and title
        ax1.set_xlabel('') # remove x label from top plot #ax.set_xlabel('Node Conformity', fontsize=12)
        ax1.set_ylabel('% Agents', fontsize=12)

        if title:
            ax1.set_title('Conformity Score Distribution',fontsize=14)
        
        if grid:
            ax1.grid(axis='y')
            
        if stat_plot:
        # Add global conformity annotation
            ax1.text(
                0.99,
                0.95,
                f"Global Degree of Conformity {label}: {global_conformity:.2f}",
                transform=ax.transAxes,
                size=12,
                horizontalalignment="right"
            )
        
        # Set x-axis limits
        ax1.set_xlim(-1.0, 1.0)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        if legend:
            title_legend = "Groups & Conformity Values" if stat else None
            ax1.legend(title=title_legend)
    
    # Bottom subplot - Overall probaboilities (Single histogram)
    for config in sub3_configs:
        label = config['label']
        global_conformity = config['conformity']
        color = config['color']
        data = config['data']
        label_plot = f"{label} (Conformity: {global_conformity:.2f})" if stat else label
        stat_plot = False # if legend else stat
        # Create histogram using seaborn
        sns.histplot(
            data=data, bins='auto',
            color=color, edgecolor = 'w', alpha=0.6,  # alternative colors: lightgreen, lightsalmon
            stat='percent', ax=ax2,  # alternative: stat='frequency'
            kde=True, # compute the kernel density estimate to smooth the distribution and show on the plot as (one or more) line(s)
            line_kws={'color': color,'linestyle':'-','linewidth':2, 'alpha': 1.0}, # 'label': 'KDE'},
            label = label_plot  # dict with Parameters that control the KDE visualization
        )
        
        # eventualmente aggiungere un secondo asse con l'altro plot (per gli agenti pessimisti)
        

        if transparent_plot_area: # plot area background
            ax2.set_facecolor('none')
        else:
            ax2.set_facecolor('white')  # default value

        # Set labels and title
        ax2.set_xlabel('') # remove x label from top plot #ax.set_xlabel('Node Conformity', fontsize=12)
        ax2.set_ylabel('% Agents', fontsize=12)
        
        title = False # remove title in this subplot
        if title:
            ax2.set_title('Conformity Score Distribution',fontsize=14)
        
        if grid:
            ax2.grid(axis='y')
            
        if stat_plot:
        # Add global conformity annotation
            ax2.text(
                0.99,
                0.95,
                f"Global Degree of Conformity {label}: {global_conformity:.2f}",
                transform=ax.transAxes,
                size=12,
                horizontalalignment="right"
            )
        
        # Set x-axis limits
        ax2.set_xlim(-1.0, 1.0)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)

        if legend:
            title_legend = "Groups & Conformity Values" if stat else None
            ax2.legend(title=title_legend)
    
    # Adjust layout
    plt.tight_layout()
    if filename is not None:
        bg_color = 'none' if transparent_bg else 'white'
        plt.savefig(filename, dpi=300, facecolor=bg_color, bbox_inches='tight')
        fig = plt.gcf()
        plt.close()
        return fig, (ax1,ax2)
    else:
        return plt.gcf(), (ax1,ax2)
        # plt.close()

# Example usage:
"""
# First compute conformity scores
node_conformity, global_conformity = compute_conformity_scores(G, y_labels, alphas=[1.0])

# Create visualization
fig, ax = plot_conformity_distribution(node_conformity, global_conformity)
plt.show()
plt.close()

# Optionally, customize the figure size
fig, ax = plot_conformity_distribution(conformity_results, figsize=(12, 8))
plt.show()
plt.close()

# Optionally visualize and compare the conformity histograms in 2x1 subplots
sub1_configs = [
        {'data': node_conformity_opt, 'conformity': global_conformity_opt, 'color': 'lightgreen', 'label': 'Optimistic'},
        {'data': node_conformity_pess,'conformity':global_conformity_pess,  'color': 'lightsalmon', 'label': 'Pessimistic'}, 
        {'data': node_conformity_neut, 'conformity': global_conformity_neut, 'color': 'lightblue', 'label': 'Neutral'}
    ]
sub3_configs = [
    {'data': node_conformity_ops, 'conformity':global_conformity_ops, 'color': 'lightgrey', 'label': 'Overall ops'}
]

fig, (ax1,ax2) = create_conformity_distribution_subplots(sub1_configs, sub3_configs,
                                filename = f"{timestamp}_conformity_subplot_ops_{simulator.model.actual_iteration}.png", ax = None, 
                                stat=True, title=True, legend=True, figure_size=(10, 8), grid=False,
                                transparent_bg=False, transparent_plot_area=False) 
"""
