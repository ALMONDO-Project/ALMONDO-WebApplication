import networkx as nx
import io
import base64
import os
import json
import datetime
import uuid
import shutil
import pandas as pd
from exceptions.custom_exceptions import ValidationError, FileUploadError, SimulationError, GraphNotFoundError, ConfigurationError
from almondo_model import AlmondoModel, OpinionDistribution, OpinionEvolution
from services.Conformity_scores import plot_conformity_distribution, create_conformity_distribution_subplots
import matplotlib.pyplot as plt

def create_simulation_directory(simulation_folder: str = 'data/simulation_results') -> tuple[str, str]:
    """
    Create a unique simulation directory with timestamp and UUID
    Args:
    - simulation_folder (str): The base folder where simulations will be stored.
    Returns:
    - sim_id (str): A unique identifier for the simulation.
    - sim_path (str): The full path to the created simulation directory.
    Raises:
    - ValidationError: If the folder path is invalid.
    - FileUploadError: If the folder cannot be created.
    """
    if not isinstance(simulation_folder, str) or not simulation_folder.strip():
        raise ValidationError("Simulation folder must be a non-empty string.", field="simulation_folder")
    
    try: 
        # Generate unique ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_id = f"{timestamp}_{str(uuid.uuid4())[:8]}"
        sim_path = os.path.join(simulation_folder, sim_id)

        # Create directory
        os.makedirs(sim_path, exist_ok=True)
        # Create plot subfolder and ensure it exists
        plots_path = os.path.join(sim_path, 'plots')
        os.makedirs(plots_path, exist_ok=True)

        return sim_id, sim_path
    
    except (OSError, IOError) as e:
        raise FileUploadError(f"Failed to create simulation directory: {e}")
    except Exception as e:
        raise ConfigurationError(f"Unexpected error creating simulation directory: {e}")

def cleanup_old_simulations(rm_method: str = 'days', simulation_folder: str = 'data/simulation_results', 
                            days_old: int = 7, size_limit: int = 50, n_save: int = 5):
    """
    Remove simulations older than specified days (or other methods) or archive them.
    Arguments:
    - rm_method (str): Method to remove old simulations:
        - 'days': Remove simulations older than a specified number of days.
        - 'size': Remove simulations based on size (in MB).
        - 'all': Remove all simulations results.
        - 'n_save': Save the last n_save simulations and remove all the others (not implemented).
    - simulation_folder (str): The folder where simulations are stored.
    - days_old (int): Number of days to consider for cleanup if rm_method is 'days'.
    Returns:
    - None
    Raises:
    - ValueError: If rm_method is not recognized.
    """
    sim_folder = simulation_folder
    # Archive old simulations
    """ # Uncomment this block if you want to archive old simulations instead of deleting them
    for sim_dir in os.listdir(sim_folder):
        sim_path = os.path.join(sim_folder, sim_dir)
        if os.path.isdir(sim_path):
            created_time = datetime.fromtimestamp(os.path.getctime(sim_path))
            if created_time < cutoff_date:
                shutil.move(sim_path, os.path.join(sim_folder, 'archive', sim_dir))
    """
    # Delete old simulations
    if os.path.exists(sim_folder):
        for sim_dir in os.listdir(sim_folder):
            sim_path = os.path.join(sim_folder, sim_dir)
            if os.path.isdir(sim_path):
                if rm_method == 'days':
                    # Check if the folder is older than the cutoff date
                    # and if it is, delete it
                    # Note: os.path.getmtime() returns the last modification time, not creation time
                    # If you want to use creation time, use os.path.getctime() instead
                    # However, getctime() may not be available on all platforms (e.g., Linux)
                    cutoff_date = datetime.datetime.now() - datetime.timedelta(days=days_old)  
                    # Get folder creation/modification time
                    folder_time = datetime.fromtimestamp(os.path.getmtime(sim_path))
                    if folder_time < cutoff_date:
                        try:
                            shutil.rmtree(sim_path)
                            print(f"Cleaned up old simulation: {sim_folder}")
                        except Exception as e:
                            print(f"Error cleaning up {sim_folder}: {e}")
                elif rm_method == 'size':
                    # Check if the folder size exceeds the specified limit
                    folder_size = sum(os.path.getsize(os.path.join(sim_path, f)) for f in os.listdir(sim_path) if os.path.isfile(os.path.join(sim_path, f)))
                    folder_size_mb = folder_size / (1024 * 1024)  # Convert to MB
                    if folder_size_mb > size_limit:
                        # If the folder size exceeds the limit, delete it
                        try:
                            shutil.rmtree(sim_path)
                            print(f"Cleaned up old simulation: {sim_folder}")
                        except Exception as e:
                            print(f"Error cleaning up {sim_folder}: {e}")
                elif rm_method == 'all':
                    # Remove all simulations
                    try:
                        shutil.rmtree(sim_path)
                        print(f"Cleaned up old simulation: {sim_folder}")
                    except Exception as e:
                        print(f"Error cleaning up {sim_folder}: {e}")
                elif rm_method == 'n_save':
                    # Save the last n_save simulations and remove all the others
                    sim_dirs = sorted([d for d in os.listdir(sim_folder) if os.path.isdir(os.path.join(sim_folder, d))])
                    for sim_dir in sim_dirs[:-n_save]:
                        sim_path = os.path.join(sim_folder, sim_dir)
                        try:
                            shutil.rmtree(sim_path)
                            print(f"Cleaned up old simulation: {sim_folder}")
                        except Exception as e:
                            print(f"Error cleaning up {sim_folder}: {e}")
                else:
                    raise ValueError(f"Unknown rm_method: {rm_method}")
    elif not os.path.exists(sim_folder):
        print("No old simulations to clean up")
        return
   

# Save graph_type_edgelist.edgelist and adjacency_matrix.csv for the download
def save_graph_files(G: nx.graph, graphType: str, upload_folder: str, 
                     save_type: str = 'edgelist' ) -> str | dict:
    """
    Save the graph as an edgelist file to the specified upload folder.

    Args:
    - G (nx.Graph): NetworkX graph object.
    - graph_type (str): Type/name of the graph (used for filename). Options are:
            'erdos_renyi', 'watts_strogatz', 'barabasi_albert', 'complete_graph', 'edgelist' or 'adjacency_matrix'
    - upload_folder (str): Directory to save the file.
    - save_type (str): One of 'edgelist', 'adjacency_matrix', or 'both'.

    Returns:
    - str or dict: Path(s) to saved file(s). Single path for one type, dict for 'both'.

    Raises:
    - ValidationError: If any input is missing or invalid.
    - FileUploadError: If the directory is not writable.
    - SimulationError: If saving the graph fails.
    """
    if G is None or not isinstance(G, nx.Graph):
        raise GraphNotFoundError("Invalid or missing graph object in save_graph_file().")

    if not graphType or not isinstance(graphType, str):
        raise ValidationError("Invalid or missing graph type in save_graph_file().", field="graph_type")

    if not upload_folder or not isinstance(upload_folder, str):
        raise ValidationError("Invalid or missing upload folder in save_graph_file().", field="upload_folder")

    if not os.path.exists(upload_folder):
        raise FileUploadError(f"Upload folder for saving graph file does not exist: {upload_folder}")

    if not os.access(upload_folder, os.W_OK):
        raise FileUploadError(f"Upload folder for saving graph file is not writable: {upload_folder}")

    if save_type not in ("edgelist", "adjacency_matrix", "both"):
        raise ValidationError("Invalid save_type for graph file. Must be 'edgelist', 'adjacency_matrix', or 'both'.", field="save_type")

    # Save based on choice
    results = {}

    try:
        if save_type in ("edgelist", "both"):
            edgelist_filename = f"{graphType}.edgelist"
            edgelist_path = os.path.join(upload_folder, edgelist_filename)
            nx.write_edgelist(G, edgelist_path, data=False)
            results["edgelist"] = edgelist_path
        if save_type in ("adjacency_matrix", "both"):
            adjacency_filename = f"{graphType}_adjacency.csv"
            adjacency_path = os.path.join(upload_folder, adjacency_filename)
            adj_matrix = nx.to_numpy_array(G, dtype=int)
            pd.DataFrame(adj_matrix).to_csv(adjacency_path, index=False, header=False)
            results["adjacency_matrix"] = adjacency_path
        
        # Return single path or dict
        if save_type == "edgelist":
            return results["edgelist"]
        elif save_type == "adjacency_matrix":
            return results["adjacency_matrix"]
        else:
            return results
        
    except Exception as e:
        raise SimulationError(f"Failed to save graph edgelist or csv: {e}")

def save_config(graph:nx.Graph, params: dict, RESULTS_FOLDER: str = 'simulation_results'):
    """
    Save the current simulation configuration to a file.

    Arguments:
    - graph (nx.Graph): The NetworkX graph representing the simulation.
    - params (dict): A dictionary containing model parameters. It includes:
        - graph_type (string)
        - graph_params (dict): a dictionary containg the parameters used to generate the graph.
        - p_o (float): The probability of opinion formation.
        - p_p (float): The probability of opinion persistence.
        - lambdaValue (float): The lambda value for the model.
        - phiValue (float): The phi value for the model.
        - lobbyists_data (dict, default={}): A dictionary containing data for the lobbyists (e.g., strategies, parameters).
        - model_seed (int, default=42): The seed for the random number generator.
        - n_lobbyists (int, default=0): The number of lobbyists in the simulation.
    - RESULTS_FOLDER (str, default='simulation_results'): The directory path where the configuration will be saved.
    Returns
    - None
    Raises:
    - ValidationError: If inputs are invalid.
    - FileUploadError: If the results folder cannot be created.
    - SimulationError: If saving files fails.
    """
    # if model is None or not isinstance(model, AlmondoModel):
    #    raise ValidationError("Invalid or missing model instance for saving model configuration.", field="model")
    
    # if not hasattr(model, 'graph'):
    #     raise ValidationError("Model must contain a valid NetworkX graph for saving configuration.", field="model.graph")

    if not isinstance(params, dict):
        raise ValidationError("Params of model configuration must be a dictionary.", field="params")

    if not isinstance(RESULTS_FOLDER, str) or not RESULTS_FOLDER.strip():
        raise ValidationError("RESULTS_FOLDER for saving model configuration must be a valid string path.", field="RESULTS_FOLDER")

    try:
        # Ensure results folder exists
        if not os.path.exists(RESULTS_FOLDER):
            os.makedirs(RESULTS_FOLDER)

        # Save graph as an edge list
        graph_path = os.path.join(RESULTS_FOLDER, 'graph.csv')
        nx.write_edgelist(graph, graph_path, delimiter=",", data=False)

        # Save the configuration to a JSON file
        config_path = os.path.join(RESULTS_FOLDER, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(params, f, indent=4)
    
    except (OSError, IOError) as e:
        raise FileUploadError(f"File system error while saving configuration: {e}")
    except TypeError as e:
        # Raised if params contains non-serializable values
        raise ValidationError(f"Parameters contain non-serializable values for saving configuration: {e}", field="params")
    except Exception as e:
        raise SimulationError(f"Unexpected error saving configuration: {e}")

# Save final results
# def save_system_status(model:AlmondoModel, path: str):
def save_system_status(system_status: dict, path: str):
    """
    Save the system status to a JSON file.

    Arguments:
    - system_status: The system status dictionary containing the simulation results.
    - path (str): The directory path where the status will be saved.
    Returns:
    - str: Full path to the saved status.json file.
    Raises:
    - ValidationError: If model or path are invalid, or system_status is missing.
    - FileUploadError: If the directory cannot be created or is not writable.
    - SimulationError: For unexpected runtime errors.

    """
    # if model is None:
    #    raise ValidationError("Model is required for saving simulation results.", field='model')

    # if not hasattr(model, 'system_status'):
    #    raise ValidationError("Model must have 'system_status' attribute for saving simulation results. " \
    #    "You must run the simulation before saving results.", field='model.system_status')

    if not isinstance(path, str) or not path.strip():
        raise ValidationError("Path for saving simulation results must be a valid non-empty string.", field="path")
    
    try:
        os.makedirs(path, exist_ok=True)

        filename = os.path.join(path, 'status.json')
        with open(filename, 'w') as f:
            json.dump(system_status, f) # json.dump(model.system_status, f)
        
        return filename

    except (OSError, IOError) as e:
        raise FileUploadError(f"File system error while saving system status of simulation: {e}")
    except TypeError as e:
        # Happens if system_status contains non-serializable objects
        raise ValidationError(f"System status of simulation contains non-serializable values: {e}", field="system_status")
    except Exception as e:
        raise SimulationError(f"Unexpected error saving system status of simulation: {e}")
