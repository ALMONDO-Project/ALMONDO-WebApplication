import os
import numpy as np
import networkx as nx
import pandas as pd
from typing import Optional
from exceptions.custom_exceptions import FileUploadError, ValidationError, GraphNotFoundError, ConfigurationError


def generate_graph_fcn(graph_type: str, form_data: dict, files: dict) -> nx.Graph:
    """
    Generate a graph based on the specified type and parameters.
    
    Args:
        graph_type (str): The type of graph to generate. Options are: 'erdos_renyi', 'watts_strogatz',
            barabasi_albert', 'complete_graph', 'edgelist' or 'adjacency_matrix'
        form_data (dict): Form data containing graph parameters. The parameters depend on the graph type:
            - 'erdos_renyi': {'nodes': int, 'prob': float, 'seed': Optional[int]}
            - 'watts_strogatz': {'nodes': int, 'k_neighbors': int, 'rewiring_prob': float, 'seed': Optional[int]}
            - 'barabasi_albert': {'nodes': int, 'm': int, 'seed': Optional[int]}
            - 'complete_graph': {'n': int}
        files (dict): Files from the request (for edgelist/adjacency matrix)
    
    Returns:
        nx.Graph: The generated graph object

    Raises:
        ValueError, TypeError, KeyError: For input-related issues
        Exception: For unexpected internal issues
    """
    def get_int(name: str, min_val: int = None) -> int:
        value = form_data.get(name)
        if value is None:
            raise ValueError(f"Missing parameter in form data in generate_graph(): '{name}'")
        try:
            value = int(value)
        except ValueError:
            raise ValidationError(f"Parameter '{name}' in in form data in generate_graph() must be an integer.") # ValueError(f"Parameter '{name}' must be an integer.")
        if min_val is not None and value < min_val:
            raise ValidationError(f"Parameter '{name}' in form data in generate_graph() must be >= {min_val}.") # valueError()
        return value

    def get_float(name: str, min_val: float = None, max_val: float = None) -> float:
        value = form_data.get(name)
        if value is None:
            raise ValueError(f"Missing parameter in form data in generate_graph(): '{name}'")
        try:
            value = float(value)
        except ValueError:
            raise ValidationError(f"Parameter '{name}' in form data in generate_graph() must be a float.") # ValueError
        if min_val is not None and value < min_val:
            raise ValidationError(f"Parameter '{name}' in form data in generate_graph() must be >= {min_val}.") # ValueError
        if max_val is not None and value > max_val:
            raise ValidationError(f"Parameter '{name}' in form data in generate_graph() must be <= {max_val}.") # ValueError
        return value

    def get_optional_seed() -> Optional[int]:
        seed = form_data.get('seed')
        if seed is None or seed == '':
            return None
        try:
            return int(seed)
        except ValueError:
            raise ValidationError("Parameter 'seed' in form data in generate_graph() must be an integer if provided.") # ValueError

    try:  
        G: Optional[nx.Graph] = None
        
        if graph_type == 'erdos_renyi':
            N = get_int('nodes', min_val=1) # int(form_data.get('nodes'))
            prob = get_float('prob', min_val=0.0, max_val=1.0) # float(form_data.get('prob'))
            seed = get_optional_seed() # form_data.get('seed', None)
            # seed = int(seed) if seed else None
            G = nx.erdos_renyi_graph(N, prob, seed=seed)
            G.graph['type'] = 'erdos_renyi'
            
        elif graph_type == 'watts_strogatz':
            N = get_int('nodes', min_val=1)
            k_neighbors = get_int('k_neighbors', min_val=1) # int(form_data.get('k_neighbors')) # Each node is joined with its k nearest neighbors in a ring topology
            if k_neighbors >= N:
                raise ValidationError("'k_neighbors' in form data in generate_graph() for watts strogatz must be less than 'nodes'") # ValueError
            rewiring_prob = get_float('rewiring_prob', min_val=0.0, max_val=1.0) # float(form_data.get('rewiring_prob')) # The probability of rewiring each edge
            seed = get_optional_seed() 
            # seed = int(seed) if seed else None
            G = nx.watts_strogatz_graph(N, k_neighbors, rewiring_prob, seed=seed)
            G.graph['type'] = 'watts_strogatz'
            
        elif graph_type == 'barabasi_albert':
            N = get_int('nodes', min_val=1)
            m = get_int('m', min_val=1) # int(form_data.get('m')) # Number of edges to attach from a new node to existing nodes
            if m >= N:
                raise ValidationError("Number of edges 'm' in form data in generate_graph()  must be less than 'nodes'") #ValueError
            seed = get_optional_seed()
            # seed = int(seed) if seed else None
            G = nx.barabasi_albert_graph(N, m, seed=seed)
            G.graph['type'] = 'barabasi_albert'
            
        elif graph_type == 'complete_graph':
            N = get_int('n', min_val=1)
            G = nx.complete_graph(N)
            G.graph['type'] = 'complete_graph'
            
        elif graph_type == 'edgelist':
            if 'uploaded_edgelist' not in files:
                raise ValidationError("No file provided in files form in generate_graph() for edgelist", field="file") # ValueError("No edgelist file uploaded.")
            edgelist_file = files['uploaded_edgelist']

            if edgelist_file.filename == '':
                raise ValidationError("No file selected in files form in generate_graph() for edgelist", field="file")
            
            try:
                G = nx.read_edgelist(edgelist_file, delimiter=' ', nodetype=int)
                if G is None or G.number_of_nodes() ==0:
                    raise GraphNotFoundError("Edgelist file in files form in generate_graph() is empty.") # ValueError
            except Exception as e:
                raise FileUploadError(f"Failed to read edgelist in files form in generate_graph(): {e}") # ValueError(f"Failed to read edgelist: {e}")
            G.graph['type'] = 'edgelist'
                
        elif graph_type == 'adjacency_matrix':
            """
            Load adjacency matrix from file or use provided NumPy array.
            """
            if 'uploaded_adjacency_matrix' not in files:
                raise ValidationError("No file provided in files form in generate_graph() for adjacency_matrix.", field="file") # ValueError("No edgelist file uploaded.") # ValueError("No adjacency matrix file uploaded.")

            adjacency_matrix_file = files['uploaded_adjacency_matrix']
            if adjacency_matrix_file.filename == '':
                raise ValidationError("No file selected in files form in generate_graph() for adjacency_matrix.", field="file")

            try: 
                df = pd.read_csv(adjacency_matrix_file, header=None)
                if df.empty:
                    raise GraphNotFoundError("Adjacency matrix file in files form in generate_graph() is empty.") # ValueError
                G = nx.from_numpy_array(df.values)
            except Exception as e:
                raise FileUploadError("No adjacency matrix file uploaded in files form in generate_graph().") # ValueError(f"Failed to read adjacency matrix: {e}")
    
            G.graph['type'] = 'adjacency_matrix'
        else:
            raise ValidationError(f"Unsupported graph type in files form in generate_graph(): {graph_type}") # ValueError

        return G
    
    except (ValueError, TypeError, ValidationError, GraphNotFoundError, FileUploadError):
        raise
    except Exception as e:
        raise ConfigurationError(f"Unexpected generation or upload error in generate_graph() (internal error): {e}")