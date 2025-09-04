from typing import Tuple, Optional
import ndlib.models.ModelConfig as mc
from almondo_model import AlmondoModel
from .lobbyists_strategies_management import create_single_random_strategy
from exceptions.custom_exceptions import ConfigurationError, ValidationError, FileUploadError
import networkx as nx
from flask import current_app
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)

def config_model(graph: nx.Graph, initial_status: list, params: dict, files: Optional[dict], sim_path: str, new_sim: bool = True) -> Tuple[Optional[AlmondoModel], dict]:
    """
    Configures the model with agent parameters, initial distribution, and graph settings.

    Arguments:
    - graph: A NetworkX graph representing the network of agents.
    - initial_status: A list representing the initial status of each agent.
    - params: A dictionary containing model parameters such as 'p_o', 'p_p', 'lambda_values', 'phi_values', 'model_seed', 'n_lobbyists', and 'lobbyists_data'.
    - files: A dictionary containing uploaded files, which may include lobbyist strategies.
    - sim_path: The path to the simulation directory where strategies will be saved.
    - new_sim: A boolean indicating whether to start a new simulation.
    Returns:
    - model: An instance of AlmondoModel configured with the provided parameters.
    - params: The updated parameters dictionary, potentially modified with lobbyist strategy paths.
    Raises:
    - ValidationError: If parameters or inputs are invalid.
    - FileUploadError: If a strategy file is invalid/unreadable.
    - SimulationError: For unexpected runtime issues.
    """
    # if not isinstance(graph, nx.Graph):
    #    raise ValidationError("Graph must be a valid NetworkX Graph.", field="graph")

    if graph is None or graph.number_of_nodes() == 0:
        raise ValidationError("Graph must be a non-empty NetworkX Graph for model configuration.", field="graph")
    
    if not isinstance(initial_status, list) or len(initial_status) != graph.number_of_nodes():
        raise ValidationError("Initial status must be a list with length equal to number of nodes for model configuration.", field="initial_status")

    if not isinstance(params, dict):
        raise ValidationError("Params for model configuration must be a dictionary.", field="params")

    if not isinstance(files, dict):
        raise ValidationError("Files for model configuration must be a dictionary.", field="files")

    if not isinstance(sim_path, str) or not sim_path.strip():
        raise ValidationError("Simulation path for model configuration must be a valid string.", field="sim_path")
    
    try: 
        config = mc.Configuration()
        # Assigning p_o and p_p parameters
        if params['p_o'] < 0.0 or params['p_o'] > 1.0:
            raise ValidationError(f"Parameter 'p_o' in config_model() must be >= 0.0 and <= 1.0.", field="p_o")
        if params['p_p'] < 0.0 or params['p_p'] > 1.0:
            raise ValidationError(f"Parameter 'p_p' in config_model() must be >= 0.0 and <= 1.0.", field="p_p")
        config.add_model_parameter("p_o", params['p_o'])
        config.add_model_parameter("p_p", params['p_p'])

        # Configure lambda values for each agent
        if isinstance(params['lambda_values'], list):
            if len(params["lambda_values"]) != graph.number_of_nodes():
                raise ValidationError("lambda_values list length in config_model() must match number of nodes.", field="lambda_values")
            if params["lambda_values"] < 0.0 or params["lambda_values"] > 1.0:
                raise ValidationError(f"Parameter 'lambda_values' in config_model() must be >= 0.0 and <= 1.0.", field="lambda_values")
            for i in graph.nodes():
                config.add_node_configuration("lambda", i, params['lambda_values'][i])
        elif isinstance(params['lambda_values'], float):
            # print('Assigning homogeneous lambda')
            if params['lambda_values'] < 0.0 or params['lambda_values'] > 1.0:
                raise ValidationError(f"Parameter 'lambda_values' in config_model() must be >= 0.0 and <= 1.0.", field="lambda_values")
            for i in graph.nodes():
                config.add_node_configuration("lambda", i, params['lambda_values'])
        else:
            raise ValidationError("lambda_values in config_model() must be float or list.", field="lambda_values")

        # Configure phi values for each agent
        if isinstance(params['phi_values'], list):
            if len(params["phi_values"]) != graph.number_of_nodes():
                raise ValidationError("phi_values list length in config_model() must match number of nodes.", field="phi_values")
            if params["phi_values"] < 0.0 or params["phi_values"] > 1.0:
                raise ValidationError(f"Parameter 'phi_values' in config_model() must be >= 0.0 and <= 1.0.", field="phi_values")
            for i in graph.nodes():
                config.add_node_configuration("phi", i, params['phi_values'][i])
        elif isinstance(params['phi_values'], float):
            # print('Assigning homogeneous phi')
            if params["phi_values"] < 0.0 or params["phi_values"] > 1.0:
                raise ValidationError(f"Parameter 'phi_values' in config_model() must be >= 0.0 and <= 1.0.", field="phi_values")
            for i in graph.nodes():
                config.add_node_configuration("phi", i, params['phi_values'])
        else:
            raise ValidationError("phi_values in config_model() must be a float or a list.", field='phi_values')

        # Initialize the model with the graph and configuration
        current_app.logger.info('Configuring model: assigning graph, parameters, and initial distribution of weights')
        model = AlmondoModel(graph, seed=params.get('model_seed', None))

        # Set the initial status in the model
        # Setting initial status in the model # TODO: Implement logging and remove print statements
        model.set_initial_status(config, kind='custom', status=initial_status)

        # Lobbyists
        n_lobbyists = int(params.get('n_lobbyists', 0))
        if new_sim:
            if n_lobbyists > 0:
                if not isinstance(params['lobbyists_data'], list) or params['lobbyists_data'] == []:
                    raise ValidationError("params['lobbyists_data'] for config_model() must be a list of dictionaries for each lobbyist.", field="lobbyists_data")
                if len(params['lobbyists_data']) != n_lobbyists:
                    raise ValidationError(f"params['lobbyists_data'] for config_model() must be a list of {n_lobbyists} dictionaries as number of lobbyists, but got {len(params['lobbyists_data'])}.", 
                                        field="lobbyists_data")
                # Manage both automatic and manual strategies for lobbyists
                # If the strategy is manual, it will be randomly generated (based on B,c,T), added to the model and saved to a file;
                # If the strategy is automatic, it will read from the strategy file, added to the model and saved into a file;
                current_app.logger.info('Assign strategies to lobbyists...')
                # The data for each lobbyist is stored in params['lobbyists_data']
                # Each lobbyist has a budget (B), model (m), and strategies (which will be populated now)
                for id in range(n_lobbyists):
                    file_key = f'lobbyist_strategy_file_{id}'
                    filename = f'{file_key}.txt'
                    data = params['lobbyists_data'][id]
                    if data is None or not isinstance(data, dict):
                        raise ValidationError("Each lobbyist's data must be a dictionary in config_model().", field=f"lobbyists_data[{id}]")
                    if not isinstance(data['m'], int) or data['m'] != 0 and data['m'] != 1:
                        raise ValidationError("Each lobbyist's model 'm' key must be an integer 0 (for pessimistic) or 1 (for optimistic) for config_model().", 
                                            field=f"lobbyists_data[{id}]")
                    m = data['m']

                    # If strategy file provided -> manual
                    if file_key in files:
                        strategy_file = files[file_key]

                        if not strategy_file or not strategy_file.filename:
                            raise FileUploadError(f"Empty or invalid strategy file for lobbyist {id} in config_model()")

                        # Read and process the CSV or txt file
                        try:
                            # Ensure the file is a valid CSV or txt file
                            if strategy_file.filename.endswith('.csv'):
                                # Read CSV content
                                csv_content = strategy_file.read().decode('utf-8')
                                # Parse CSV to matrix using pandas or csv module
                                import pandas as pd
                                from io import StringIO
                                
                                # Option A: Using pandas
                                df = pd.read_csv(StringIO(csv_content), header=None)
                                strategy_matrix = df.values.tolist()  # Convert to list of lists
                                
                                # Option B: Using csv module
                                # import csv
                                # reader = csv.reader(StringIO(csv_content))
                                # strategy_matrix = [[int(cell) for cell in row] for row in reader]
                            elif strategy_file.filename.endswith('.txt'):
                                # Read TXT content
                                txt_content = strategy_file.read().decode('utf-8')
                                # Parse TXT content into strategy matrix
                                strategy_matrix = [[int(cell) for cell in line.split()] for line in txt_content.splitlines()]

                            else:
                                raise ValidationError(f"Strategy file for lobbyist {id} in config_model() must be .csv or .txt.", field="strategy_file")
                            
                            strategy_matrix = np.array(strategy_matrix)  # Convert to numpy array

                            # Check if the strategy matrix matches the number of agents in the graph
                            if np.shape(strategy_matrix)[1] != graph.number_of_nodes():
                                raise ValidationError(
                                    f"In config_model() strategy matrix for lobbyist {id} does not match the number of agents in the graph. Expected {graph.number_of_nodes()} columns, got {np.shape(strategy_matrix)[1]}.",
                                    field="strategy_matrix"
                                )
                            if not np.all(np.isin(strategy_matrix, [0, 1])):
                                raise ValidationError(f"In config_model() strategy matrix for lobbyist {id} contains values other than 0 and 1.", 
                                                    field="strategy_matrix")

                            # Add lobbyist with strategy to the model
                            model.add_lobbyist(m, strategy_matrix)
                            # Store the matrix and the other data in the lobbyist data
                            B = int(strategy_matrix.sum())  # Assuming budget B is the sum of the elements in the strategy matrix
                            params['lobbyists_data'][id]['B'] = B
                            params['lobbyists_data'][id]['T'] = np.shape(strategy_matrix)[0]  # Assuming active time steps are the number of rows
                            params['lobbyists_data'][id]['c'] = 1 # Assuming cost is 1 for simplicity

                            # save the strategy to a file
                            # print(f'Saving strategy of lobbyist {id} to file')
                            folder = os.path.join(sim_path, 'strategies', str(B))
                            os.makedirs(folder, exist_ok=True)
                            path = os.path.join(folder, filename)
                            np.savetxt(path, strategy_matrix, fmt="%i")
                            # print(f'Strategy matrix of lobbyist {id} saved to file')
                            # Store the path to the strategy file in the lobbyist data
                            params['lobbyists_data'][id]['strategies'] = [path]

                        except (TypeError, ValueError) as e:
                            raise ConfigurationError(f"Configuration error while adding lobbyist {id} in config_model(): {e}")
                        
                        except (OSError, IOError) as e:
                            raise FileUploadError(f"File error while processing lobbyist {id} in config_model(): {e}")
                        
                        except Exception as e:
                            raise ConfigurationError(f"Unexpected error in manual strategy for lobbyist {id} in config_model(): {e}")

                    else:
                        # Automatic strategy for lobbyist, generated based on B, c, T parameters
                        # print(f'No strategy file found for lobbyist {id}, creating random strategy')
                        # Create a random strategy matrix
                        try:
                            if 'B' not in data or 'T' not in data or 'c' not in data:
                                raise ValidationError(f"Missing B, T, or c in lobbyist {id} data for automatic strategy for config_model().", 
                                                    field=f"lobbyists_data[{id}]")
                            if isinstance(data['B'], int) is False or data['B'] < 0:
                                raise ValidationError(f"Budget 'B' for lobbyist {id} must be a non negative integer for config_model().", field=f"lobbyists_data[{id}]['B']")
                            if isinstance(data['T'], int) is False or data['T'] < 0:
                                raise ValidationError(f"Time steps 'T' for lobbyist {id} must be a non negative integer for config_model().", field=f"lobbyists_data[{id}]['T']")
                            if isinstance(data['c'], int) is False or data['c'] < 0:
                                raise ValidationError(f"Cost 'c' for lobbyist {id} must be a non negative integer for config_model().", field=f"lobbyists_data[{id}]['c']")
                            
                            B = data['B'] # Total budget of lobbyist
                            T = data['T'] # Total time steps
                            c = data['c'] # Cost per signal
                            N = graph.number_of_nodes()  # Number of agents in the graph
                            strategy_matrix = create_single_random_strategy(B, T, N, c)
                            # Add lobbyist with strategy to the model
                            model.add_lobbyist(m, strategy_matrix)
                            # save the strategy to a file
                            # print(f'Saving strategy of lobbyist {id} to file')
                            folder = os.path.join(sim_path, 'strategies', str(B))
                            os.makedirs(folder, exist_ok=True)
                            path = os.path.join(folder, filename)
                            np.savetxt(path, strategy_matrix, fmt="%i")
                            # current_app.logger.info(f'Strategy matrix of lobbyist {id} saved to file')
                            # Store the matrix and the other data in the lobbyist data
                            params['lobbyists_data'][id]['strategies'] = [path]

                        except (TypeError, ValueError) as e:
                            raise ConfigurationError(f"Configuration error while adding lobbyist {id} in config_model(): {e}")
                        
                        except Exception as e:
                            raise ConfigurationError(f"Error creating random strategy for lobbyist {id} in config_model(): {e}")
                current_app.logger.info('Lobbyists strategies created')
            else:
                current_app.logger.info('No lobbyists to assign strategies to.')
        else:
            if n_lobbyists > 0:
                if not isinstance(params['lobbyists_data'], list) or params['lobbyists_data'] == []:
                        raise ValidationError("params['lobbyists_data'] for config_model() must be a list of dictionaries for each lobbyist.", field="lobbyists_data")
                if len(params['lobbyists_data']) != n_lobbyists:
                    raise ValidationError(f"params['lobbyists_data'] for config_model() must be a list of {n_lobbyists} dictionaries as number of lobbyists, but got {len(params['lobbyists_data'])}.", 
                                        field="lobbyists_data")
                    # Assigning strategies to lobbyists
                for id in range(n_lobbyists):
                    strategy_file = params['lobbyists_data'][id].get('strategies', [None])[0]
                    txt_content = strategy_file.read().decode('utf-8')
                    # Parse TXT content into strategy matrix
                    strategy_matrix = [[int(cell) for cell in line.split()] for line in txt_content.splitlines()]
                    strategy_matrix = np.array(strategy_matrix)  # Convert to numpy array

                    # Check if the strategy matrix matches the number of agents in the graph
                    if np.shape(strategy_matrix)[1] != graph.number_of_nodes():
                        raise ValidationError(
                            f"In config_model() strategy matrix for lobbyist {id} does not match the number of agents in the graph. Expected {graph.number_of_nodes()} columns, got {np.shape(strategy_matrix)[1]}.",
                            field="strategy_matrix"
                        )
                    if not np.all(np.isin(strategy_matrix, [0, 1])):
                        raise ValidationError(f"In config_model() strategy matrix for lobbyist {id} contains values other than 0 and 1.", 
                                            field="strategy_matrix")

                    # Add lobbyist with strategy to the model
                    model.add_lobbyist(m, strategy_matrix)
        
        return model, params
    
    except ValidationError:
        raise
    except FileUploadError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Unexpected error in config_model: {e}")