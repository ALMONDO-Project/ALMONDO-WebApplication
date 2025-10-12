from almondo_model.classes.almondoModel import AlmondoModel
from almondo_model.functions import metrics_functions
import networkx as nx
import numpy as np
from almondo_model.functions.utils import transform
from services.Conformity_scores import probabilities_clustering, compute_conformity_scores, compute_conformity_scores_opinion
from typing import Optional
from flask import current_app
from exceptions.custom_exceptions import GraphNotFoundError, ConfigurationError, ValidationError, MetricsError
import logging

logger = logging.getLogger(__name__)
def graph_basic_metrics(graph: nx.Graph) -> dict:
    """ Calculate basic metrics for a given graph.

    Args:
        graph (networkx.Graph): The input graph.
    Returns:
        dict: A dictionary (graph_metrics) containing basic graph metrics.
            - 'numNodes': Number of nodes in the graph.
            - 'numEdges': Number of edges in the graph.
            - 'degree': Average degree of the graph.
            - 'density': Density of the graph.
            - 'degree_hist': Degree histogram of the graph.
            - 'top_five_nodes': Top five nodes by degree centrality and their degree values.
    Raises:
        GraphNotFoundError: If input is invalid.
        SimulationError: If metrics calculation fails.
    """
    if graph is None:
        raise GraphNotFoundError("No graph found in graph_basic_metrics() computation.")
    
    if not isinstance(graph, nx.Graph):
        raise ValidationError("In graph_basic_metrics() computation, input must be a networkx Graph object.", field="graph")

    try: 
        numNodes = graph.number_of_nodes()
        numEdges = graph.number_of_edges()
        degree = dict(graph.degree()) # Get degree for each node
        avgDegree = sum(degree.values()) / numNodes if numNodes > 0 else 0
        # top five nodes by degree centrality and their degree values
        top_five_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
        
        density = nx.density(graph)
        degree_histogram = nx.degree_histogram(graph)

        """
        dict_degree = {}
        for d_h in range(len(degree_histogram)):
            dict_degree[d_h] = degree_histogram[d_h]
        """
        graph_metrics = {
            'numNodes': numNodes,
            'numEdges': numEdges,
            'degree': avgDegree,
            'density': density,
            'degree_hist': degree_histogram,
            'top_five_nodes': top_five_nodes
        }
        return graph_metrics
    
    except Exception as e:
        raise MetricsError(f"Error calculating basic graph metrics: {e}")

def get_node_info(graph: nx.Graph, system_status: Optional[dict], prior_prob: Optional[dict], node_id: int=0,  it: Optional[int]=-1, betweeness: bool=False) -> dict:
    """
    Get information about a specific node in the graph, including its degree centrality,
    betweenness centrality, and opinion in the specified iteration.
    Args:
    - graph (networkx.Graph): The input graph.
    - system_status (dict): The dictionary containing system status of simulation (optional).
    - node_id (int): The ID of the node to retrieve information for.
    - prior_prob (dict): The optimistic 'p_o' and pessimistic 'p_p' probabilities parameters for the model (optional).
    - it (int): The iteration number to retrieve the opinion for.
    - betweeness (bool): Default is False. Set to True to compute the betweeness centrality of the node.
    Returns:
        dict: A dictionary (node_info) containing node information.
            - 'node_id': The ID of the node.
            - 'node_degree_centrality': Degree centrality of the node.
            - 'node_betweenness_centrality': Betweenness centrality of the node.
            - 'opinion': The opinion of the node in the specified iteration if model is specified.
            - 'label': The label of the node based on its opinion ('optimistic', 'pessimistic', 'neutral').
            - 'iteration': The iteration number, if model is specified.
    Raises:
        ValidationError: If input arguments are invalid.
        GraphNotFound: If the graph or node is missing.
        MetricsError: If metric computation fails.
        ConfigurationError: If model/iteration configuration is invalid.
    """
    # Check input validation
    if graph is None:
        raise GraphNotFoundError("Graph is required but not provided to get_node_info().")

    if not isinstance(graph, nx.Graph):
        raise ValidationError("Input must be a networkx Graph object to get_node_info().", field="graph")
    
    if node_id is None or not isinstance(node_id, int):
         raise ValidationError("Node ID must be a valid integer to get_node_info().", field = 'node_id')

    if node_id not in list(graph.nodes):
        raise GraphNotFoundError(f"Node ID {node_id} not found in the graph to get_node_info().")
    
    try:
        # degree centrality for the specific node
        degree_centrality = nx.degree_centrality(graph)
        degree_centrality_node = degree_centrality[node_id]
    
        node_info = {
            'node_id': node_id,
            'node_degree_centrality': degree_centrality_node,
            'node_betweenness_centrality': None,
            'opinion': None,
            'label': None,
            'iteration': None
            }
        # betweeness centrality for the specific node
        
        if betweeness: # if required
            try:
                betweenness_centrality = nx.betweenness_centrality(graph)
                node_info['node_betweenness_centrality'] = betweenness_centrality[node_id]
            except Exception as e:
                raise MetricsError(f"Failed to compute betweenness centrality for node {node_id} in get_node_info(): {e}")
        
        # If model is not provided, return basic metrics only
        # if model is None:
        if system_status is None:
            return node_info # "Model not configured or not initialized." --> try except
        params = { 'p_o': prior_prob['p_o'], 'p_p': prior_prob['p_p'] }
        print('Node info:')
        print('op', prior_prob['p_o'])
        print('pp', prior_prob['p_p'])
        if not isinstance(it, int) or it < -1 or it >= len(system_status):
            raise ConfigurationError(f"Invalid iteration index in get_node_info(): {it}")
        
        # Check if node exists in model status
        status_snapshot = system_status[it]
        if str(node_id) not in status_snapshot['status']:
            # node_info['iteration'] = model.system_status[it]['iteration'] + 1
            raise GraphNotFoundError(f"Node ID {node_id} not found in the model status for iteration {it} in get_node_info().")

        # Get the opinion of the node in the specified iteration
        # weight = status_snapshot['status'][str(node_id)]
        # node_info['opinion'] = transform(weight, params['p_o'], params['p_p']) # opt model
        
        # The status received from frontend has probabilities
        node_info['opinion'] = status_snapshot['status'][str(node_id)]

        if node_info['opinion'] <= 0.33:
            label = 'optimistic'
        elif node_info['opinion'] > 0.66:
            label = 'pessimistic'
        else:
            label = 'neutral'
        node_info['label'] = label
        node_info['iteration'] = status_snapshot['iteration'] + 1  # Return the iteration number as well

        return node_info
    
    except (ValidationError, GraphNotFoundError, ConfigurationError, MetricsError):
        raise
    except Exception as e:
        raise MetricsError(f"Unexpected error in get_node_info: {e}")

def get_opinion_statistics(system_status: dict, prior_prob: dict, it: int=-1) -> dict:
    """
    Get statistics about the opinions in the graph for a specific iteration
    divided by optimistic agents (opinion > 0.66), neutral agents (0.33 < opinion <= 0.66), 
    and pessimistic agents (opinion <= 0.33). 
    Args:
        system_status (dict): The system status dictionary containing agent opinions.
        prior_prob (dict): The optimitic 'p_o' and pessimistic 'p_p' probabilities for the model.
        it (int, optional): The iteration number to retrieve statistics for. Defaults to -1 (latest iteration).
    Returns:
        - dict: A dictionary (opinion_diffusion_statistics) containing statistics for optimistic, neutral, and pessimistic agents.
            - optimistic_agents/pessimistic_agents/neutral_agents dictionaries: Statistics for each set of agents containing:
                - 'mean': Mean opinion of optimistic agents.
                - 'std': Standard deviation of opinions of optimistic agents.
                - 'num_agents': Number of optimistic agents.
                - 'percentage': Percentage of optimistic agents in the graph.
    Raises:
        ConfigurationError: If model is missing or system_status invalid.
        GraphNotFound: If graph is missing or empty.
        ValidationError: If iteration index is invalid.
        MetricsError: If calculation of statistics fails.       
    """
    # if model is None:
    #     raise ConfigurationError("Error to get_opinion_statistics(): model is not initialized.")
    
    # if not hasattr(model, "system_status") or not model.system_status:
    if not system_status:
        raise ConfigurationError("Error to get_opinion_statistics(): Model system_status is missing or empty.")

    if not isinstance(it, int):
        raise ValidationError("Error to get_opinion_statistics(): Iteration index must be an integer.", field="iteration")
    if it < -1 or it >= len(system_status):
        raise ValidationError(
            f"Error to get_opinion_statistics(): Invalid iteration specified. It must be between -1 and {len(system_status) - 1}.", 
            field='iteration')
    
    # Get number of agents in the graph
    N = len(system_status[-1]['status'])  # model.graph.number_of_nodes()
    if N == 0:
        raise GraphNotFoundError("Error to get_opinion_statistics(): Graph is empty.")
    
    try:
        # Get weights of the nodes in the specified iteration and transform it into subjective probabilities
        ops = system_status[it]['status'].values()
        weights = np.array([el for el in ops])
        probabilities = prior_prob['p_o'] * weights + prior_prob['p_p'] * (1 - weights)  # Optimistic model

        # Classify optimistic, neutral, and pessimistic agents probabilities
        pess_probabilities = probabilities[probabilities > 0.66]
        neutral_probabilities = probabilities[(probabilities <= 0.66) & (probabilities > 0.33)]
        opt_probabilities = probabilities[probabilities <= 0.33]

        # Compute stats 
        def stats(values, total_agents):
            if len(values) == 0:
                return {'mean': 0, 'std': 0, 'num_agents': 0, 'percentage': 0}
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'num_agents': len(values),
                'percentage': (len(values) / total_agents) * 100 if total_agents > 0 else 0,
            }
        optimistic_agents_stats = stats(opt_probabilities, N)
        neutral_agents_stats = stats(neutral_probabilities, N)
        pessimistic_agents_stats = stats(pess_probabilities, N)

        # Prepare the response data
        return {
            'optimistic_agents': optimistic_agents_stats,
            'neutral_agents': neutral_agents_stats,
            'pessimistic_agents': pessimistic_agents_stats
        }
    except (ValidationError, ConfigurationError, GraphNotFoundError):
        raise
    except Exception as e:
        raise MetricsError(f"Unexpected error calculating opinion statistics: {e}")    

def calculate_opinion_metrics(system_status:dict, prior_prob: dict, it: int=-1, lobb_models_list: list=[]) -> dict:
    """
    Calculate opinion metrics for the given model and iteration.
    - overall mean and standard deviation of opinion distribution
    - number of iterations
    - current iteration the metrics referred to
    - effective number of clusters of opinions
    - pairwise absolute distance of opinions
    - if any lobbyist in the simulation: performance index for each supported model calculated
    as mean of agents subjective probabilities – prior probability of supported model (p_o or p_p)
    Args:
        - system_status (dict): The dictionary of simulation results
        - prior_prob (dict): The optimitic 'p_o' and pessimistic 'p_p' probabilities for the model.
        - it (int, optional): The iteration number to retrieve metrics for. Defaults to -1 (latest iteration).
        - lobb_models_list (list): list of lobbyists models
    Returns:
        - dict: A dictionary (opinion_metrics) containing statistics for agents opinions in iteration it.
            - 'opinion_mean': Mean opinion of all agents.
            - 'opinion_std': Standard deviation of opinions of all agents.
            - 'opinion_pairwise_distance': max absolute pairwise distance among agents.
            - 'number_clusters': the effective nunumber_clusters of agents opinions (probabilities).
            - 'number_iterations': number of iterations of the whole simulation,
            - 'current_iteration': iteration the statistics are referred to.
            - 'lobbyists_performance': a dictiorany containg the lobbyists_performance index for each supported model
                ('pessimistic' or 'optimistic'), defined as mean of prob – prior probability of supported model (p_o or p_p)
    Raises:
        ConfigurationError: If the model or system_status is not properly set.
        ValidationError: If the iteration index is invalid.
        MetricsError: If calculation fails.
    """
    # Inputs validation
    # if model is None:
    #     raise ConfigurationError("Error to calculate_opinion_metrics(): model is not initialized.")
    
    # if not hasattr(model, "system_status") or not model.system_status:
    if not system_status:
        raise ConfigurationError("Error to calculate_opinion_metrics(): Model system_status is missing or empty.")

    if not isinstance(it, int):
        raise ValidationError("Error to calculate_opinion_metrics(): Iteration index must be an integer.", field="iteration")
    if it < -1 or it >= len(system_status):
        raise ValidationError(
            f"Error to calculate_opinion_metrics(): Invalid iteration specified. It must be between -1 and {len(system_status) - 1}.", 
            field='iteration')
    try:
        # Get weights of the nodes in the specified iteration and transform it into subjective probabilities
        ops = system_status[it]['status'].values()
        weights = np.array([el for el in ops])
        probabilities = prior_prob['p_o'] * weights + prior_prob['p_p'] * (1 - weights)  # Optimistic model

        # Calcutate mean, standard deviation of opinions, and pairwise absolute distance
        ops_mean = probabilities.mean().tolist()
        ops_std = probabilities.std().tolist()
        ops_pairwise_distance = metrics_functions.pwdist(probabilities).tolist()

        # Effective Number of clusters
        number_clusters = metrics_functions.nclusters(probabilities,0.0001)

        # number of iterations
        num_iterations = system_status[-1]['iteration'] + 1 # last iteration
        current_iteration = system_status[it]['iteration'] # current iteration

        lobbyists_performance = {'optimistic': None, 'pessimistic': None} # imitialize the dictionary

        # if any lobbyists calculate the performance index for each model
        if len(lobb_models_list) > 0:
            # extract lobbyists models
            # model_list = [l.m for l in model.lobbyists]
            lob_models = list(set(lobb_models_list)) # unique values from previous set
            # Calculate lobbyists performance index for each supported model (0 for pessimistic, 1 for optimistic)
            # as mean of prob – prior probability of supported model (p_o or p_p)
            lobb_opt_performance = abs(ops_mean - prior_prob['p_o'])
            lobb_pess_performance = abs(ops_mean - prior_prob['p_p'])
            # create the 
            if len(lob_models)==1:
                if 0 in lob_models:
                    lobbyists_performance['pessimistic'] = lobb_pess_performance
                elif 1 in lob_models:
                    lobbyists_performance['optimistic'] = lobb_opt_performance
                else:
                    raise ConfigurationError("Found unsupported lobbyist model in calculate_opinion_metrics().")
            elif len(lob_models) == 2:
                lobbyists_performance['pessimistic'] = lobb_pess_performance
                lobbyists_performance['optimistic'] = lobb_opt_performance

        opinion_metrics = {
            'opinion_mean': ops_mean,
            'opinion_std': ops_std,
            'opinion_pairwise_distance': ops_pairwise_distance,
            'number_clusters': number_clusters,
            'number_iterations': num_iterations,
            'current_iteration': current_iteration,
            'lobbyists_performance': lobbyists_performance
        }

        return opinion_metrics

    except (ValueError, ValidationError, ConfigurationError):
        raise
    except Exception as e:
        raise MetricsError(f"Unexpected error calculating opinion metrics: {e}")

def calculate_conformity_scores(graph: nx.Graph, system_status: dict, prior_prob: dict, it: int=-1, 
                                mode: str = 'prob_cluster', dist_threshold: float = 0.01) -> dict:
    """
    Compute conformity scores for each node in a graph and the overall conformity degree
    based on based on automatic (aggomerative) clustering  on agents probabilities
    or optimist (prob<=0.33), pessimist (p>0.66) or neutral (other probabilities) label of nodes.
    
    Args:
        - graph (networkx.Graph): The input graph.
        - system_status (dict): The dictionary of the simulation results.
        - prior_prob (dict): The optimitic 'p_o' and pessimistic 'p_p' probabilities for the model.
        - it (int, optional): The iteration number to retrieve metrics for. Defaults to -1 (latest iteration).
        - mode (str): 'prob_clusters' or 'op_label' or 'both'
        if 'prob_clusters' mode is selected the algorithm computes an agglomerative clustering over the probabilities
        of the iteration it of the mode. 
            In this case the use can select the dist_threshold parameter. If this param is 0.05,
            the dataset is divided in no more than 20 clusters; if it is 0.01, the datatset is divided 
            in maximum 50 clusters.
        If 'ops_label' mode is selected, the algorithm computes the conformity with respect to the node labels
        'optimistic' (agent probability <=0.33), 'pessimistic' (agent probability >0.66) 
         or 'neutral' (agent probability in (0.33, 0.66])
        If 'both' mode is selected both conformity measures are computed
    Returns a 
    - dict: A dictionary (opinion_metrics) containing conformity scores for agents opinions in iteration it.
        - 'prob_clusters': A dict containing 'node_conformity' dictionary and 'global_conformity' for probability clustering.
        - 'ops_label': A dict containing four dictionaries for the conformity computed with the 'ops_label' method 
        with respect to the node labels ('opt', 'pess', 'neutral').
        Each dictionary has the same structure of 'prob_clusters' dictionary with fields 'node_conformity' and 'global_conformity'.
            - 'overall': the overall conformity of the whole graph with respect to the nodel labels 
            - 'optimistic': conformity scores of only optimistic agents
            - 'pessimistic': conformity scores of only pessimistic agents
            - 'neutral': conformity scores of only neutral agents.
    Raises:
        ConfigurationError: If the model or system_status is not properly set.
        ValidationError: If the iteration index is invalid.
        MetricsError: If calculation fails.
    Notes:
    ------
    The conformity score measures how similar a node is to its neighbors in terms of 
    cluster assignments. A higher score indicates greater conformity with neighboring nodes.
    """
    # Check input validation
    if graph is None:
        raise GraphNotFoundError("Graph is required but not provided to calculate_conformity_scores().")

    if not isinstance(graph, nx.Graph):
        raise ValidationError("Input must be a networkx Graph object to calculate_conformity_scores().", field="graph")

    #if model is None:
    #    raise ConfigurationError("Error to calculate_conformity_scores(): model is not initialized.")
    
    # if not hasattr(model, "system_status") or not model.system_status:
    if not system_status: 
        raise ConfigurationError("Error to calculate_conformity_scores(): Model system_status is missing or empty.")

    if not isinstance(it, int):
        raise ValidationError("Error to calculate_conformity_scores(): Iteration index must be an integer.", field="iteration")
    if it < -1 or it >= len(system_status):
        raise ValidationError(
            f"Error to calculate_conformity_scores(): Invalid iteration specified. It must be between -1 and {len(system_status) - 1}.", 
            field='iteration')
    if not isinstance(mode, str) or mode not in ['prob_clusters', 'ops_label', 'both']:
        raise ValidationError("Error to calculate_conformity_scores(): Invalid mode specified. It must be one of ['prob_clusters', 'ops_label', 'both'].", field='mode')

    try:
        # Get weights of the nodes in the specified iteration and transform it into subjective probabilities
        ops = system_status[it]['status'].values()
        weights = np.array([el for el in ops])
        probabilities =prior_prob['p_o'] * weights + prior_prob['p_p'] * (1 - weights)  # Optimistic model
    except Exception as e:
        raise MetricsError(f"Error retrieving probabilities for iteration {it} for calculate_conformity_scores(): {e}")

    node_conformity_dict = {'prob_clusters': {}, 'ops_label': {}}
    # Probability clustering conformity 
    if mode in ['prob_clusters', 'both']:
        try: 
            current_app.logger.info('Clusterizing probabilities for conformity score...')
            _, y_label, _, _, _ = probabilities_clustering(probabilities, 
                                                            cluster_mode='auto', 
                                                            clust_algorithm='average', 
                                                            dist_threshold=0.01)
            # time_clustering = t1-t0
        # print(f'Time for clustering agents opinion: {time_clustering} s')
        except (ValidationError, MetricsError):
            raise
        except Exception as e:
            raise MetricsError(f"Error while clusterizing probabilities or computing conformity scores: {str(e)}")

        try:
            # First compute conformity scores
            current_app.logger.info('Computing conformity scores...')
            node_conf, global_conf = compute_conformity_scores(G=graph, y_label=y_label, alphas=[1.0])
            # Create dictionary for output
            node_conformity_dict['prob_clusters']={
            'node_conformity': node_conf,
            'global_conformity': global_conf
            }
        except Exception as e:
                raise MetricsError(f"Error while computing conformity scores: {str(e)}")

    # Opinion-label conformity 
    if mode in ['ops_label', 'both']:
        try:
            node_conf_ops, global_conf_ops = compute_conformity_scores_opinion(G=graph, probabilities=probabilities, alphas=[1.0])
            # Partition by opinion category
            node_conf_opt =  {int(i): node_conf_ops[i] for i in np.where(probabilities > 0.33)[0]} #node_conf_ops[np.where(probabilities > 0.33)[0]]
            # global_conformity_opt = sum(node_conformity_opt.values()) / len(node_conformity_opt)
            node_conf_pess = {int(i): node_conf_ops[i] for i in np.where(probabilities > 0.66)[0]}
            # global_conformity_pess = sum(node_conformity_pess.values()) / len(node_conformity_pess)
            node_conf_neut =  {int(i): node_conf_ops[i] for i in np.where((probabilities <= 0.66) & (probabilities > 0.33))[0]} #node_conformity_ops[(probabilities <= 0.66) & (probabilities > 0.33)]
            # global_conformity_neut = sum(node_conformity_neut.values()) / len(node_conformity_neut)
       
            # Create dictionaries for output
            node_conformity_dict['ops_label'] = {
                'overall': {
                    'node_conformity': node_conf_ops,
                    'global_conformity': global_conf_ops,
                },
                'optimistic': {
                    'node_conformity': node_conf_opt,
                    'global_conformity': (sum(node_conf_opt.values()) / len(node_conf_opt)) if node_conf_opt else 0,
                },
                'neutral': {
                    'node_conformity': node_conf_neut,
                    'global_conformity': (sum(node_conf_neut.values()) / len(node_conf_neut)) if node_conf_neut else 0,
                },
                'pessimistic': {
                    'node_conformity': node_conf_pess,
                    'global_conformity': (sum(node_conf_pess.values()) / len(node_conf_pess)) if node_conf_pess else 0,
                },
            }

        except Exception as e:
                raise MetricsError(f"Error while computing opinion label conformity scores: {str(e)}")
  
    return node_conformity_dict