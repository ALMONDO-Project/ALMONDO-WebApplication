import os
from flask import Flask, current_app, render_template, request, jsonify, send_file, Response
from config import config
from exceptions.custom_exceptions import AppError, FileUploadError, GraphNotFoundError, ValidationError, SimulationError, ConfigurationError, MetricsError
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import networkx as nx
from services.simulation_manager import SimulationManager
from services.file_manager import create_simulation_directory, cleanup_old_simulations 
from services.file_manager import save_graph_files, save_config, save_system_status
from services.plots_generator import save_final_plots, save_conformity_plots
from services.graph_generator import generate_graph_fcn
from services.initial_status_generator import generate_initial_status
from services.model_configuration import config_model
from services.metrics_and_statistics import graph_basic_metrics, get_node_info, get_opinion_statistics, calculate_opinion_metrics, calculate_conformity_scores
from flask_cors import CORS
import json
import sys
import csv
import copy

# You can set the environment using in the terminal:
#  export FLASK_CONFIG=production
# or
# export FLASK_CONFIG=development

import logging # --> commentare se non si vuole il logging
from logging.handlers import RotatingFileHandler  # --> commentare

def configure_logging(app):  # --> commentare
    if not app.debug and not app.testing:
        # File handler for production
        file_handler = RotatingFileHandler(
            'logs/app.log', maxBytes=10240000, backupCount=5
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s '
            '[in %(pathname)s:%(lineno)d] -- %(funcName)s'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Application startup')

def create_app(config_name=None):
    """Application factory pattern"""
    app = Flask(__name__)
    
    # Load configuration
    config_name = config_name or os.environ.get('FLASK_CONFIG') or 'default'
    app.config.from_object(config[config_name])
    
    # Initialize app with configuration
    config[config_name].init_app(app)
    
    # Configure logging for development --> commentare
    if app.config['DEBUG']:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),  # This ensures output goes to terminal
                logging.FileHandler('debug.log')
            ]
        )
        app.logger.setLevel(logging.DEBUG)
    # app.simulations = SimulationManager()
    return app

# Create app instance
app = create_app(config_name='development')
CORS(app)
configure_logging(app)  # 
logger = logging.getLogger(__name__) # 

from routes.error_handlers import register_error_handlers
register_error_handlers(app)

# Function to check allowed file types
def allowed_file(filename):
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_EXTENSIONS'])

@app.route('/')
def index():
    return render_template('index.html')

# Example route showing how to access config
@app.route('/debug-config')
def debug_config():
    return {
        'upload_folder': current_app.config['UPLOAD_FOLDER'],
        'results_folder': current_app.config['RESULTS_FOLDER'],
        'generated_graphs': current_app.config['GENERATED_GRAPHS_FOLDER']
    }

# Global variables
Graph = None
model = None

"""
@app.route('/<sim_id>/plots/<filename>')  # Add this route to serve plot files from the simulation directory for static approach
def serve_plot(sim_id, filename):
    # Serve plot files from the data/simulation_results/{sim_id}/plots directory
    try:
        # Security validations
        if not filename:
            current_app.logger.warning("Missing filename parameter in serving plot.")
            raise ValidationError("Missing filename parameter in serving plot.")

        # Prevent directory traversal attacks
        if '..' in filename or '/' in filename or '\\' in filename:
            current_app.logger.warning("Invalid filename parameter in serving plot.")
            raise ValidationError("Invalid filename parameter in serving plot.")

        # Only allow image files
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.svg']
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            current_app.logger.warning("Invalid file type for serving plot.")
            raise ValidationError("Invalid file type for serving plot.")

        # Construct the full path
        plot_path = os.path.join(current_app.config['RESULTS_FOLDER'], sim_id, 'plots', filename)
        current_app.logger.info(f"Serving plot file: {plot_path}")

        # Check if file exists
        if not os.path.exists(plot_path):
            current_app.logger.warning(f"Plot file not found: {plot_path}")
            raise ValidationError("Plot file not found.")

        # Determine mimetype
        if filename.lower().endswith('.png'):
            mimetype = 'image/png'
        elif filename.lower().endswith('.jpg') or filename.lower().endswith('.jpeg'):
            mimetype = 'image/jpeg'
        elif filename.lower().endswith('.svg'):
            mimetype = 'image/svg+xml'
        else:
            mimetype = 'application/octet-stream'

        return send_file(plot_path, mimetype=mimetype)

    except Exception as e:
        current_app.logger.error(f"Unexpected serving plot error: {e}")
        return jsonify({
            'success': False,
            'message': f"Serving plot {filename} for frontend failed due to server error.",
            'error': 'internal_server_error'}), 500
"""
    
@app.route('/generate-graph', methods=['POST'])
def generate_graph():
    """
    Generate graph and return graph data to frontend
    Form data from frontend containing graph parameters. The parameters depend on the graph type:
    - graph_type (str): Options are: 'erdos_renyi', 'watts_strogatz', 'barabasi_albert', 'complete_graph', 'edgelist' or 'adjacency_matrix'
            - 'erdos_renyi': {'nodes': int, 'prob': float, 'seed': Optional[int]}
            - 'watts_strogatz': {'nodes': int, 'k_neighbors': int, 'rewiring_prob': float, 'seed': Optional[int]}
            - 'barabasi_albert': {'nodes': int, 'm': int, 'seed': Optional[int]}
            - 'complete_graph': {'n': int}
    files (dict): Files from the request (for 'edgelist'/'adjacency_matrix' options)
    """
    form_data = request.form.to_dict()
    files = request.files
    graphType = request.form.get('graphType')
    if not graphType:
        raise ValidationError("Missing 'graph_type' in form data for generate_graph().")
    
    global Graph
    
    # TODO: add 'seed' for reproducibility in the frontend
    # Call the service function
    Graph = generate_graph_fcn(graph_type=graphType,
        form_data=form_data, files=files)

    # session_id = current_app.simulations.create(Graph)
    # Save the graph files --> togliere se si vuole lasciare che sia solo l'utente a salvarlo esplicitamente. è salvato in automatico al termine della simulazione
    save_graph_files(Graph, graphType, current_app.config['GENERATED_GRAPHS_FOLDER'])

    # Convert graph to D3 format (nodes and links)
    nodes = [{'id': str(node)} for node in Graph.nodes()]
    links = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in Graph.edges()]
    # current_app.logger.info(f'Graph generated: {Graph.graph["type"]}')
    # current_app.logger.info(f'List of nodes: {list(Graph.nodes())}... total {Graph.number_of_nodes()} nodes')
    # current_app.logger.info(f'List of links: {list(Graph.edges())}... total {Graph.number_of_edges()} edges')  
    response = {
            'success': True,
            'message': "Graph generated successfully.",
            'nodes': nodes,
            'links': links,
            'num_nodes': Graph.number_of_nodes(),
            'num_edges': Graph.number_of_edges(),
            'graph_type': graphType,
            #'session_id': session_id
        } 
    current_app.logger.info(f"Graph generated successfully.")
    return jsonify(response)

@app.route('/download-edge-list', methods=['GET'])
def download_edge_list():
    # TODO: eliminare
    # global Graph
    # Graph.graph['type'] is always consistent because the edgelist can only be downloaded if the graph has been recently generated
    # Validate that a graph exists
    # data = json.request
    # session_id = data['simulation_id']
    # graph = current_app.simulations.get_graph(session_id)
    """
    Example if list of nodes and edges are retrieved from frontend data form
    Graph = nx.Graph() # nx.DiGraph() for directed graphs
    Graph.add_nodes_from(form_data.get('nodes', []))
    Graph.add_edges_from(form_data.get('edges', []))
    Graph.graph['type'] = form_data.get('graph_type', 'erdos_renyi') # 'erdos_renyi' or other types
    """

    if Graph is None or not hasattr(Graph, 'graph') or 'type' not in Graph.graph:
        raise ValidationError("No graph available for download as an edgelist.", field='graph')

    # Build file path
    upload_folder = current_app.config.get('GENERATED_GRAPHS_FOLDER')
    if not upload_folder:
        raise FileUploadError("Download graphs folder is not configured.")

    try:
        filepath = os.path.join(upload_folder, f"{Graph.graph['type']}.edgelist")

        # Validate file existence
        if not os.path.exists(filepath):
            raise GraphNotFoundError(f"Graph file not found: {filepath}")

        if not os.access(filepath, os.R_OK):
            raise FileUploadError(f"Graph file is not readable: {filepath}")

        current_app.logger.info(f"Graph file is ready for download: {filepath}")
        # Serve file
        return send_file(filepath, as_attachment=True)
    
    except (GraphNotFoundError, FileUploadError):
        raise
    except (Exception, OSError) as e:
        raise ConfigurationError(f"Unexpected download error in download_edge_list(): {e}")

@app.route('/download-matrix', methods=['GET'])
def download_matrix():
    # TODO: eliminare
    # global Graph
    # Graph.graph['type'] is always consistent because the adjmatrix can only be downloaded if the graph has been recently generated
    # data = json.request
    # session_id = data['simulation_id']
    # graph = current_app.simulations.get_graph(session_id)
    # Validate Graph
    """
    If data_form from frontend provides list of nodes and edges of the graph, the last can be used to create the graph.
    Graph = nx.Graph() # nx.DiGraph() for directed graphs
    Graph.add_nodes_from(form_data.get('nodes', []))
    Graph.add_edges_from(form_data.get('edges', []))
    Graph.graph['type'] = form_data.get('graph_type', 'erdos_renyi') # 'erdos_renyi' or other types
    """
    if Graph is None or not hasattr(Graph, 'graph') or 'type' not in Graph.graph:
        raise ValidationError("No graph available for download as an adjacency matrix.", field="graph")

    graph_type = Graph.graph['type']

    # --- Validate Upload Folder ---
    upload_folder = current_app.config.get('GENERATED_GRAPHS_FOLDER')
    if not upload_folder:
        raise FileUploadError("Download graphs folder is not configured.")

    if not os.path.exists(upload_folder):
        raise FileUploadError(f"Download graphs folder does not exist: {upload_folder}")

    if not os.access(upload_folder, os.W_OK):
        raise FileUploadError(f"Download graphs folder is not writable: {upload_folder}")

    adjacency_matrix_file = os.path.join(upload_folder, f"{graph_type}_adjacency_matrix.csv")

    # Save Matrix
    try:
        adj_matrix = nx.to_pandas_adjacency(Graph, dtype=int)
        if adj_matrix.empty:
            raise ValidationError("Generated adjacency matrix is empty.", field="matrix")

        adj_matrix.to_csv(adjacency_matrix_file, index=False, header=False)

        current_app.logger.info(f"Adjacency matrix file is ready for download: {adjacency_matrix_file}")
        # --- Return File ---
        return send_file(adjacency_matrix_file, as_attachment=True)
    except ValidationError:
        raise
    except Exception as e:
        raise ConfigurationError(f"Failed to generate adjacency matrix: {e}")

@app.route('/load-simulation', methods=['POST'])
def load_simulation():
    """ Accepts the 'sim_id' parameter from frontend: this parameter specifies the id of the simulation to load"""
    sim_id = request.form['sim_id']
    sim_path = os.path.join(current_app.config['RESULTS_FOLDER'], sim_id)

    edges_list = []

    with open(sim_path + '/graph.csv', mode='r') as f:
        edges = csv.reader(f)
        for edge in edges:
            edges_list.append((int(edge[0]), int(edge[1])))
    
    # Recreate graph and assing it to global variable
    global Graph
    global model

    Graph = nx.Graph()
    Graph.add_edges_from(edges_list)
    Graph.graph['type'] = 'edgelist'

    nodes = [{'id': str(node)} for node in Graph.nodes()]
    links = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in Graph.edges()]

    # Retrieve system status and model parameters

    with open(sim_path + '/status.json', 'r') as f:
        system_status = json.load(f)
    
    with open(sim_path + '/config.json', 'r') as f:
        params = json.load(f)
    
    initial_status = list(system_status[-1]['status'].values())

    # Model Configuration
    m, params = config_model(
        graph=Graph, initial_status=initial_status,
        params=params, sim_path=sim_path, files={}, new_sim=False
    )

    m.system_status = system_status
    m.actual_iteration = system_status[-1]['iteration']

    model=m

    transformed_sys_status = copy.deepcopy(m.system_status)

    for status in transformed_sys_status:
        it_status = status['status']

        for agent in list(it_status.keys()):
            it_status[agent] = params['p_o'] * it_status[agent] + params['p_p'] * (1 - it_status[agent])

    return jsonify(
        {
            'success': True,
            'nodes': nodes,
            'links': links,
            'sim_params': params,
            'sim_results': transformed_sys_status,
            'simulation_id': sim_id
        }
    )

@app.route('/simulations-ids')
def get_simulations_IDs():
    return os.listdir(current_app.config['RESULTS_FOLDER'])

@app.route('/run-simulation', methods=['POST'])
def run_simulation():
    """
    Form data from frontend containing graph data (nodes, edges, graph_type) and the simulation parameters.
    - 'po': optimistic model probability,
    - 'pp': pessimistic model probability,
    - 'initialStatus': initial status type. Options are ('uniform', 'unbiased', 'gaussian_mixture', 'user_defined') ,
    - 'lambdaValue': float or list of floats of the under-reaction parameter.
    - 'phiValue': float or list of floats of the directional motivated reasoning parameter.
    - 'modelSeed': seed for reproducibility
    - 'n_lobbyists': number of lobbyists in the simulation. Default to 0 if not specified
    - 'lobbyists_data': Dictionary with data of each lobbyist. 
    - 'runSimulationOption': Options for running the simulation: 'iteration-bunch' or 'steady-state'
    - 'iterations': Number of iterations to run the simulation for 'iteration-bunch' option
    Files:
    - 'status': intial status file path (csv or txt) for 'user-defined' option
    - lobbyists strategy files for manual option
    """
    # Retrieve the global variables
    # global Graph, model
    """
    If data_form from frontend provides list of nodes and edges of the current graph, the last can be used to create the graph.
    Graph = nx.Graph() # nx.DiGraph() for directed graphs
    Graph.add_nodes_from(form_data.get('nodes', []))
    Graph.add_edges_from(form_data.get('edges', []))
    Graph.graph['type'] = form_data.get('graph_type', 'erdos_renyi') # 'erdos_renyi' or other types
    """
    global model  # TODO: eliminare se si passa lo stato e grafo direttamnete da frontend e si ricostruisce il modello
    form_data = request.form.to_dict()
    files = request.files 
    # Clean up old simulations before running new one
    # cleanup_old_simulations(days_old=7, simulation_folder=current_app.config['RESULTS_FOLDER'])
    sim_id, sim_path = create_simulation_directory(current_app.config['RESULTS_FOLDER'])
    # format_plot_output = 'base64'  # Default format for plots, can be changed to 'png' if needed (static plots)
    # session_id = form_data['simulation_id']
    # Graph = current_app.simulations.get_graph(session_id)
    # current_app.simulations.update_sim_id(session_id, sim_id)
    # 
    # Retrieve parameters from the form
    runSimulationOption = (form_data.get('runSimulationOption'))
    params = {
        'p_o': float(form_data.get('po',0.01)),
        'p_p': float(form_data.get('pp',0.99)),
        'initialStatus_type': (form_data.get('initialStatus')),
        'its': 0,  # number of iterations (updated if iteration_bunch option is selected). This is the default value, can be changed in the frontend
        'lambda_values': float(form_data.get('lambdaValue')),
        'phi_values': float(form_data.get('phiValue')),
        'model_seed': int(form_data.get('modelSeed', 42)),  # Default seed is 42
        'n_lobbyists': int(form_data.get('n_lobbyists', 0)),  # Default to 0 if not specified
        'lobbyists_data': json.loads(form_data.get('lobbyists_data', '[]'))  # Default to empty list of dictionaries if not specified
    }
    # Validate parameters
    if params['p_o'] < 0 or params['p_o'] > 1:
        raise ValidationError(f"Parameter 'p_o' in form data for running simulation must be >= 0.0 and <= 1.0.", field="p_o")
        
    
    if params['p_p'] < 0 or params['p_p'] > 1:
        raise ValidationError(f"Parameter 'p_p' in form data for running simulation must be >= 0.0 and <= 1.0.", field="p_p")

    if not params['initialStatus_type']:
        raise ValidationError("Initial status type not specified in form data for running simulation", field="initialStatus_type")

    if params['lambda_values'] < 0.0 or params['lambda_values'] > 1.0:
        raise ValidationError(f"Parameter 'lambda_values' in form data for running simulation must be >= 0.0 and <= 1.0.", field="lambda_values")

    if params['phi_values'] < 0.0 or params['phi_values'] > 1.0:
        raise ValidationError(f"Parameter 'phi_values' in form data for running simulation must be >= 0.0 and <= 1.0.", field="phi_values")

    if params['n_lobbyists'] < 0:
        raise ValidationError("Number of lobbyists cannot be negative in form data for running simulation", field="n_lobbyists")


    if Graph is None:
        raise ConfigurationError("Graph not generated! Please generate a graph before running a simulation...")
    
    # Generate initial status
    initial_status = generate_initial_status(
        initial_status_type=params['initialStatus_type'],
        form_data=form_data,
        files=files.to_dict(),
        num_nodes=Graph.number_of_nodes()  # Pass the number of nodes
    )

    if initial_status is None:
        raise ConfigurationError("Initial status generation failed: initial_status variable is empty.")

    current_app.logger.info("Initial status generated successfully.")

    # Model Configuration
    current_app.logger.info('Configuring model...')
    model, params = config_model(
        graph=Graph, initial_status=initial_status,
        params=params, files=files.to_dict(), 
        sim_path=sim_path
    )

    if model is None:
        raise ConfigurationError("Model configuration generation failed: model variable is empty.")

    # current_app.simulations.update_model(session_id, model)
    current_app.logger.info('Model configured successfully.')

    # Simulation execution
    try: 
        if(runSimulationOption == "iteration-bunch"):
            simulation_iterations = int(request.form.get('iterations', 100))  # Default to 100 if not provided
            params['its'] = simulation_iterations  # Update its with the number of iterations
            model.iteration_bunch(simulation_iterations)
        elif(runSimulationOption == "steady-state"):
            model.steady_state(drop_evolution=False)  # Run the steady state analysis: 
            #REMARK: if drop_evolution is True, only the final state will be saved, otherwise all evolution data will be saved.
            params['its'] = model.system_status[-1]['iteration']  # Get the last iteration number from the system status
        else:
            raise ValidationError(f"Unknown simulation option: {runSimulationOption}")
        # current_app.simulations.update_model(session_id, model)
    except MemoryError:
            raise SimulationError("Simulation requires too much memory. Try reducing the number of agents.")
        
    except OSError as e:
        raise SimulationError(f"System error during simulation: {e}")
    
    except ValueError as e:
        raise SimulationError(f"Simulation error - Simulation does not properly end: an error occurred. Details: {e}")
   
    except Exception as e:
        raise SimulationError(f"Simulation error - An unexpected error occurred during simulation: {e}")

    # Save configuration to a JSON file
    current_app.logger.info('Saving configuration...')
    save_config(graph=model.graph, params=params, RESULTS_FOLDER=sim_path)

    # Save final result
    current_app.logger.info('Saving results...')
    save_system_status(system_status=model.system_status, path=sim_path)

    # Transform weights into probabilities for each iteration of system status
    transformed_sys_status = copy.deepcopy(model.system_status)

    for status in transformed_sys_status:
        # print('inside loop')
        it_status = status['status']

        for agent in list(it_status.keys()):
            it_status[agent] = params['p_o'] * it_status[agent] + params['p_p'] * (1 - it_status[agent])
            


    return jsonify(
        {
            'success': True,
            'message': 'Simulation run successfully.',
            'sim_params': params,
            'sim_results': transformed_sys_status,
            'simulation_id': sim_id
        }
    )
    
@app.route('/generate-simulation-plots', methods=['POST'])
def generate_simulation_plots():
    """
    Routes to generate simulation plots.
    You can use this route to generate plots from the simulation results or save them.
    Form data from frontend containing graph data (nodes, edges, graph_type) and the simulation parameters and results:
    - 'po': optimistic model probability,
    - 'pp': pessimistic model probability,
    - 'system_status': dictionary with the simulation results until now
    - 'simulation_id': unique identifier for the simulation
    - 'save': True if the user wants to save the plots, otherwise the function should return the plots without saving.
    """
    # generate plots
    data = request.json
    # session_id = data['session_id']
    # graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)
    
    # Get sim_id from frontend request
    # sim_id = current_app.simulations.get_sim_id(session_id)
    sim_id = data['simulation_id']
    save = bool(int(data.get('save', 0)))  # flag variable to check if save plots (1) or send to frontend (0)
    system_status = json.loads(data.get('system_status', []))
    params = {
        'p_o': data.get('po', 0.01),
        'p_p': data.get('pp', 0.99),
    }
    format_plot_output = 'base64' if not save else 'png'  # Default format for plots, can be changed to 'png' if needed (static plots)
    
    current_app.logger.info('Generating plots...')
    if save:
        simulation_folder = current_app.config['RESULTS_FOLDER']
        sim_path = os.path.join(simulation_folder, sim_id)
        sim_plots_path = os.path.join(sim_path, 'plots')
        os.makedirs(sim_plots_path, exist_ok=True)
    if format_plot_output == 'png':  # Save plots in PNG format
        if not save:
            simulation_folder = current_app.config['RESULTS_FOLDER']
            sim_path = os.path.join(simulation_folder, sim_id)
            sim_plots_path = os.path.join(sim_path, 'plots')
            os.makedirs(sim_plots_path, exist_ok=True)
        plot_dict = save_final_plots(system_status=system_status, params=params, PLOT_FOLDER=sim_plots_path, format='png')
        opinion_plot_url = f'{sim_id}/{plot_dict["opinion_plot_url"]}'
        final_results_plot_url = f'{sim_id}/{plot_dict["final_results_plot_url"]}'
        response = {
            'success': True,
            'msg': 'Simulation plots generated and saved successfully.',
            'opinion_plot_url': opinion_plot_url,
            'final_results_plot_url': final_results_plot_url,
            'simulation_id': sim_id
        }
        
    elif format_plot_output == 'base64':
        plot_dict = save_final_plots(system_status=system_status, params=params, format='base64')
        response = {
            'success': True,
            'msg': 'Simulation plots generated successfully.',
            'opinion_plot_url': plot_dict["opinion_plot_url"],
            'final_results_plot_url': plot_dict["final_results_plot_url"],
            'simulation_id': sim_id
        }
      
    elif format_plot_output == 'svg':
        plot_dict = save_final_plots(system_status=system_status, params=params, format='svg')
        response = {
            'success': True,
            'msg': 'Simulation plots generated successfully.',
            'opinion_plot_svg': plot_dict["opinion_plot_svg"],
            'final_results_plot_svg': plot_dict["final_results_plot_svg"],
            'simulation_id': sim_id
        }
    else:
        raise ValidationError(f'Format plot type {format_plot_output} for simulation plots not valid. Use "png", "base_64" or "svg".')
    
    return jsonify(response)

@app.route('/continue-simulation', methods=['POST'])
def continue_simulation(): 
    """
    Form data from frontend containing graph data (nodes, edges, graph_type) and the simulation parameters and results:
    - 'po': optimistic model probability,
    - 'pp': pessimistic model probability,
    - 'lambdaValue': float or list of floats of the under-reaction parameter.
    - 'phiValue': float or list of floats of the directional motivated reasoning parameter.
    - 'modelSeed': seed for reproducibility
    - 'n_lobbyists': number of lobbyists in the simulation. Default to 0 if not specified
    - 'lobbyists_data': Dictionary with data of each lobbyist. 
    - 'runSimulationOption': Options for running the simulation: 'iteration-bunch' or 'steady-state'
    - 'iterations': Number of iterations to run the simulation for 'iteration-bunch' option
    - 'system_status': dictionary with the simulation results until now
    - 'sim_id': unique identifier for the simulation
    """
    # Use the global graph variable
    # global model # TODO: da eliminare se si passa direttamente lo stato e si ricostruisce il modello
    # global Graph

    runSimulationOption = request.form['runSimulationOption']
    iterations = int(request.form['iterations'])
    # Get sim_id from frontend request
    # sim_id = current_app.simulations.get_sim_id(session_id)
    sim_id = request.form['simulation_id']
    sim_path = os.path.join(current_app.config['RESULTS_FOLDER'], sim_id)
    # format_plot_output = 'base64'  # Default format for plots, can be changed to 'png' if needed (static plots)
    
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValidationError(f"Number of 'iterations' must be an integer >= 0.", field="iterations")
    # session_id = data['session_id']
    # graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)
    
    current_app.logger.info(f"Continuing simulation for {iterations} iterations for simulation id {sim_id}...")
    
    if not sim_id:
        raise ValidationError(f"Continue simulation error: simulation_id from form data is required for continuing simulation.", field="simulation_id")
    
    edges_list = []

    with open(sim_path + '/graph.csv', mode='r') as f:
        edges = csv.reader(f)
        for edge in edges:
            edges_list.append((int(edge[0]), int(edge[1])))
    
    # Recreate graph
    Graph = nx.Graph()
    Graph.add_edges_from(edges_list)
    Graph.graph['type'] = 'edgelist'

    # Retrieve system status and model parameters
    with open(sim_path + '/status.json', 'r') as f:
        system_status = json.load(f)
    
    with open(sim_path + '/config.json', 'r') as f:
        params = json.load(f)
    
    initial_status = list(system_status[-1]['status'].values())

    # Model Configuration
    model, params = config_model(
        graph=Graph, initial_status=initial_status,
        params=params, sim_path=sim_path, files={}, new_sim=False
    )

    model.system_status = system_status
    model.actual_iteration = system_status[-1]['iteration']
    """
    If data_form from frontend provides list of nodes and edges of the current graph, the last can be used to create the graph.
    Graph = nx.Graph() # nx.DiGraph() for directed graphs
    Graph.add_nodes_from(form_data.get('nodes', []))
    Graph.add_edges_from(form_data.get('edges', []))
    Graph.graph['type'] = form_data.get('graph_type', 'erdos_renyi') # 'erdos_renyi' or other types
    """
    if Graph is None:
        raise ConfigurationError("Graph not generated! Please generate a graph before continuing a simulation...")

    # system_status = json.loads(data.get('system_status', []))
    # params = {
    #     'p_o': float(data.get('po',0.01)),
    #     'p_p': float(data.get('pp',0.99)),
    #     'initialStatus_type': (data.get('initialStatus')),
    #     'its': int(system_status[-1]['iteration']),  # number of iterations (updated if iteration_bunch option is selected). This is the default value, can be changed in the frontend
    #     'lambda_values': float(data.get('lambdaValue')),
    #     'phi_values': float(data.get('phiValue')),
    #     'model_seed': int(data.get('modelSeed', 42)),  # Default seed is 42
    #     'n_lobbyists': 0, # int(data.get('n_lobbyists', 0)),  to avoid overwriting of strategies
    #     'lobbyists_data': json.loads(data.get('lobbyists_data', []))  # Default to empty list of dictionaries if not specified
    # }
    # Validate parameters
    # if params['p_o'] < 0 or params['p_o'] > 1:
    #     raise ValidationError(f"Parameter 'p_o' in form data for running simulation must be >= 0.0 and <= 1.0.", field="p_o")
        
    
    # if params['p_p'] < 0 or params['p_p'] > 1:
    #     raise ValidationError(f"Parameter 'p_p' in form data for running simulation must be >= 0.0 and <= 1.0.", field="p_p")

    # if not params['initialStatus_type']:
    #     raise ValidationError("Initial status type not specified in form data for running simulation", field="initialStatus_type")

    # if params['lambda_values'] < 0.0 or params['lambda_values'] > 1.0:
    #     raise ValidationError(f"Parameter 'lambda_values' in form data for running simulation must be >= 0.0 and <= 1.0.", field="lambda_values")

    # if params['phi_values'] < 0.0 or params['phi_values'] > 1.0:
    #     raise ValidationError(f"Parameter 'phi_values' in form data for running simulation must be >= 0.0 and <= 1.0.", field="phi_values")

    # if params['n_lobbyists'] < 0:
    #     raise ValidationError("Number of lobbyists cannot be negative in form data for running simulation", field="n_lobbyists")

    
    # # Generate initial status
    # initial_status = system_status[-1]['status']

    # if initial_status is None:
    #     raise ConfigurationError("Initial status generation failed: initial_status variable is empty.")

    # # Model Configuration
    # model, params = config_model(
    #     graph=Graph, initial_status=initial_status,
    #     params=params, sim_path=sim_path, new_sim=False
    # )

    # model.system_status = system_status  # Set the existing system status to the model
    # model.actual_iteration = system_status[-1]['iteration']

    # if model is None:
    #     raise ConfigurationError("Model configuration generation failed: model variable is empty.")

    # current_app.simulations.update_model(session_id, model)

    # TODO add tqdm bar in the frontend to show the progress of the simulation
    # Check if the model is already configured
    if model is None:
        raise ConfigurationError("Model not found: 'model' variable is empty. Impossible to continue simulation.")

    try: 
        # Simulation execution
        if(runSimulationOption == "iteration-bunch"):
            model.iteration_bunch(iterations)
        elif(runSimulationOption == "steady-state"):
            model.steady_state(drop_evolution=False)  # Run the steady state analysis: 
            #REMARK: if drop_evolution is True, only the final state will be saved, otherwise all evolution data will be saved.
        else:
            raise ValidationError(f"Unknown simulation option: {runSimulationOption}")
    
        # current_app.simulations.update_model(session_id, model)
    except MemoryError:
            raise SimulationError("Simulation requires too much memory. Try reducing the number of agents.")
        
    except OSError as e:
        raise SimulationError(f"System error during simulation: {e}")
    
    except ValueError as e:
        raise SimulationError(f"Simulation error - Simulation does not properly end: an error occurred. Details: {e}")
   
    except Exception as e:
        raise SimulationError(f"Simulation error - An unexpected error occurred during simulation: {e}")
  
    

    current_app.logger.info('Saving configuration...')
    try:
        # update the config.json file with the new number of iterations
        filename = os.path.join(sim_path, 'config.json')
        with open(filename, "r") as f:  # read the file
            params = json.load(f)
        params["its"] = model.system_status[-1]['iteration']
    
    except (OSError, json.JSONDecodeError) as e:
        raise FileUploadError(f"Failed to read configuration file for continuing simulation: {e}")
    save_config(graph=model.graph, params=params, RESULTS_FOLDER=sim_path)
    
    # Save final result
    current_app.logger.info('Saving results...')
    save_system_status(system_status=model.system_status, path=sim_path)

    transformed_sys_status = copy.deepcopy(model.system_status)

    for status in transformed_sys_status:
        it_status = status['status']

        for agent in list(it_status.keys()):
            it_status[agent] = params['p_o'] * it_status[agent] + params['p_p'] * (1 - it_status[agent])
    
    return jsonify({
            'success': True,
            'sim_params': params,
            'message': f'simulation {sim_id} successfully continued',
            'sim_results': transformed_sys_status,
            'simulation_id': sim_id
        })

@app.route('/basic-info-graph', methods=['POST'])
def get_basic_info_graph():
    """
    Get basic info about the graph
    Create a function to get basic info about the graph
    This function will return the number of nodes, number of edges, degree centrality, density, and degree histogram
    Note: Degree centrality is the fraction of nodes a node is connected to
    Density is the ratio of the number of edges to the number of possible edges
    Degree histogram is a list of the number of nodes with each degree

    The form data from the frontend provides a list of nodes and edges of the current graph, which can be used to create the graph.
    """

    # global Graph
    # data = request.get_json(silent=True)
    # session_id = data['session_id']
    # Graph = current_app.simulations.get_graph(session_id)
    # form_data = request.get_json(silent=True)
    graph_type = request.form.get('graph_type')
    nodes = list(map(int, json.loads(request.form.get('nodes'))))
    edges = list(map(lambda edge: (int(edge[0]), int(edge[1])), json.loads(request.form.get('edges'))))

    # print("form data:\n", form_data)
    # if not form_data:
    #     raise ValidationError("Request body in getOpinionDiffusionStatistics() must be valid JSON.", field="body")
    # try:
    # if form_data.get('nodes') is not None and form_data.get('edges') is not None:
    if nodes is not None and edges is not None:
        Graph = nx.Graph() # nx.DiGraph() for directed graphs
        # Graph.add_nodes_from(form_data.get('nodes', []))
        # Graph.add_edges_from(form_data.get('edges', []))
        # Graph.graph['type'] = form_data.get('graph_type', 'erdos_renyi') # 'erdos_renyi' or other types
        Graph.add_nodes_from(nodes)
        Graph.add_edges_from(edges)
        Graph.graph['type'] = graph_type
    else:
        if Graph is None:
            raise ConfigurationError("Graph not generated! Please generate a graph before getting basic info about the graph...")

    graph_basic_info = graph_basic_metrics(graph=Graph)
    
    current_app.logger.info("Basic graph metrics calculated successfully.")
    return jsonify({'success': True, 
                'message': 'Basic graph metrics calculated successfully.', 
                'graph_basic_info': graph_basic_info}), 200

    
@app.route('/basic-info-node', methods=['POST'])
def get_basic_info_node():
    """
    Get basic info about a specific node in the graph
    The form data from the frontend provides 
    - 'node_id'
    - 'iteration'
    - 'betweeness' as 1/0 flag if compute betweeness or not.
    - 'system_status' containing the status of the system at each iteration
    - 'params' containing the optimistic and pessimistic probabilities
    - 'simulation_id' containing the ID of the simulation
    """
    # global Graph, model
    request_data = request.get_json(silent=True)
    if not request_data:
        raise ValidationError("Request body to get basic info about a node must be valid JSON.", field="body")
    try:
        node_id = int(request_data.get('node_id', 0))  # Default to 0 if not specified
    except (TypeError, ValueError):
        raise ValidationError("node_id for basic_info_node() must be an integer.", field="node_id")   
    try:    
        it = int(request_data.get('iteration', -1))  # Default to last (-1) if not specified
    except (TypeError, ValueError):
        raise ValidationError("iteration for basic_info_node() must be an integer.", field="iteration")
    try:
        betweeness = bool(int(request_data.get('betweeness', 0)))  # flag variable to check if compute betweeness or not Default 0 = False
    except (TypeError, ValueError):
        raise ValidationError("betweeness flag in basic_info_node() must be 0 or 1.", field="betweeness")

    graph_type = request_data.get('graph_type')
    nodes = list(map(int, json.loads(request_data.get('nodes'))))
    edges = list(map(lambda edge: (int(edge[0]), int(edge[1])), json.loads(request_data.get('edges'))))

    # Recreating Graph
    Graph = nx.Graph()
    Graph.add_nodes_from(nodes)
    Graph.add_edges_from(edges)
    Graph.graph['type'] = graph_type
    
    if 'system_status' in request_data.keys():
        system_status = request_data['system_status']
        params = {
            'p_o': float(request_data.get('p_o', 0.01)),
            'p_p': float(request_data.get('p_p', 0.99))
        }
        sim_id = request_data.get('simulation_id')

        node_info = get_node_info(graph=Graph, system_status=system_status, prior_prob=params,
                                node_id=node_id, it=it, betweeness=betweeness)
        response = {'success': True,
                    'message': f'Basic context info for node {node_id} fetched successfully.',
                    'node_info': node_info,
                    'simulation_id': sim_id}
    else:
        node_info = get_node_info(graph=Graph, node_id=node_id, it=it,betweeness=betweeness)
        response = {'success': True,
                    'message': f'Basic context info for node {node_id} fetched successfully.',
                    'node_info': node_info}
    # session_id = request_data['session_id']
    # Graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)

    current_app.logger.info("Basic node metrics calculated successfully.")

    # Return the node info as a JSON response
    return jsonify(response), 200

@app.route('/opinion-diffusion-statistics', methods=['POST'])
def get_opinion_diffusion_statistics():
    """
    Get opinion diffusion statistics in the graph for a specific iteration
    divided by optimistic agents (opinion > 0.66), neutral agents (0.33 < opinion <= 0.66), 
    and pessimistic agents (opinion <= 0.33)
       Get basic info about a specific node in the graph
    The form data from the frontend provides 
    - 'system_status' containing the simulation results
    - 'it' iteration
    - 'p_o' containing the optimistic probability
    - 'p_p' containing the pessimistic probability
    - 'simulation_id' containing the ID of the simulation
    """
    # global model
    request_data = request.get_json(silent=True)
    if not request_data:
        raise ValidationError("Request body in getOpinionDiffusionStatistics() must be valid JSON.", field="body")
    try:
        it = int(request_data.get('iteration', -1))  # Default to last (-1) if not specified
    except (TypeError, ValueError):
        raise ValidationError("iteration in getOpinionDiffusionStatistics() must be an integer.", field="iteration")
    try:
        system_status = json.loads(request_data['system_status'])
    except (TypeError, ValueError):
        raise ValidationError("system_status in getOpinionDiffusionStatistics() must be valid JSON.", field="system_status")
    try:
        po = float(request_data.get('p_o', 0.01))
        pp = float(request_data.get('p_p', 0.99))
    except (TypeError, ValueError):
        raise ValidationError("p_o and p_p in getOpinionDiffusionStatistics() must be a float.", field="p_o")

    sim_id = request_data['simulation_id']


    # session_id = request_data['session_id']
    # Graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)
    params = { 'p_o': po, 'p_p': pp }

    opinion_diffusion_statistics = get_opinion_statistics(system_status=system_status, prior_prob=params, it=it)
    current_app.logger.info("Opinion diffusion statisticss calculated successfully.")
    # Return the opinion diffusion statistics as a JSON response
    return jsonify({'success': True,
                    'msg': f'Opinion diffusion statistics calculated successfully (iteration={it}).',
                    'opinion_diffusion_statistics': opinion_diffusion_statistics,
                    'sim_id': sim_id}), 200

@app.route('/opinion-metrics', methods=['POST'])
def get_opinion_metrics():
    """
    Get opinion metrics for a specific iteration in the graph
    The form data from the frontend provides 
    - 'system_status' containing the simulation results
    - 'it' iteration
    - 'p_o' containing the optimistic probability
    - 'p_p' containing the pessimistic probability
    - 'n_lobbyists': 0, # int(data.get('n_lobbyists', 0)),  to avoid overwriting of strategies
    - 'lobbyists_data': json.loads(data.get('lobbyists_data', [])) 
    - 'simulation_id' containing the ID of the simulation
    """
    # global model, Graph
    request_data = request.get_json(silent=True)
    if not request_data:
        raise ValidationError("Request body in getOpinionMetrics() must be valid JSON.", field="body")
    try:
        it = int(request_data.get('iteration', -1))  # Default to last (-1) if not specified
    except (TypeError, ValueError):
        raise ValidationError("iteration in getOpinionMetrics() must be an integer.", field="iteration")
    try:
        system_status = json.loads(request_data['system_status'])
    except (TypeError, ValueError):
        raise ValidationError("system_status in getOpinionDiffusionStatistics() must be valid JSON.", field="system_status")
    try:
        po = float(request_data.get('p_o', 0.01))
        pp = float(request_data.get('p_p', 0.99))
    except (TypeError, ValueError):
        raise ValidationError("p_o and p_p in getOpinionDiffusionStatistics() must be a float.", field="p_o")

    sim_id = request_data['simulation_id']
    if 'lobbyist_data' in request_data.to_dict().keys():
        lobbyists_data = json.loads(request_data['lobbyists_data'])
        lobb_models = [l.m for l in lobbyists_data]
    else:
        lobb_models = []
    # session_id = request_data['session_id']
    # Graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)
    params = {'p_o':po, 'p_p':pp}

    opinion_metrics = calculate_opinion_metrics(system_status=system_status, prior_prob=params, it=it, lobb_models_lists=lobb_models)
    current_app.logger.info("Opinion metrics calculated successfully.")
    # Return the opinion metrics as a JSON response
    return jsonify({'success': True,
                    'msg': f'Opinion metrics calculated successfully (iteration={it}).',
                    'opinion_metrics': opinion_metrics,
                    'simulation_id': sim_id}), 200

@app.route('/conformity-score', methods=['POST'])
def conformity_score():
    """
    Calculate the conformity scores and generate the plot
    # Vedi test_simulation per il flow
    The form data from the frontend provides the nodes and edges for the graph and  
    - 'system_status' containing the simulation results
    - 'it' number of iteration where calculate the conformity score
    - 'p_o' containing the optimistic probability
    - 'p_p' containing the pessimistic probability
    - 'simulation_id' containing the ID of the simulation
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
    """
    # global model, Graph
    request_data = request.get_json()
    if not request_data:
        raise ValidationError("Request body in ConformityScore() must be valid JSON.", field="body")
    try:
        it = int(request_data.get('iteration', -1))  # Default to last (-1) if not specified
    except (TypeError, ValueError):
        raise ValidationError("iteration in ConformityScore() must be an integer.", field="iteration")

    try:
        system_status = request_data['system_status']
    except (TypeError, ValueError):
        raise ValidationError("system_status in getOpinionDiffusionStatistics() must be valid JSON.", field="system_status")
    try:
        po = float(request_data.get('p_o', 0.01))
        pp = float(request_data.get('p_p', 0.99))
    except (TypeError, ValueError):
        raise ValidationError("p_o and p_p in getOpinionDiffusionStatistics() must be a float.", field="p_o")
    
    sim_id = str(request_data.get('simulation_id'))
    params = {'p_o':po, 'p_p':pp}
    nodes = list(map(int, request_data.get('nodes', [])))
    edges = list(map(lambda edge: (int(edge[0]), int(edge[1])), request_data.get('edges', [])))
    # session_id = request_data['session_id']
    # Graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)
    # sim_id = current_app.simulations.get_sim_id(session_id)
    Graph = nx.Graph() # nx.DiGraph() for directed graphs
    Graph.add_nodes_from(nodes)
    Graph.add_edges_from(edges)
    format_plot_output = 'base64'  # Default format for plots, can be changed to 'png' if needed (static plots)
        
    if not sim_id:
        raise ValidationError(f"Conformity score error: sim_id from form data is required for saving conformity score.", field="simulation_id")

    # mode for conformity scores: 'prob_clusters' or 'ops_label' or 'both'
    """
    - if 'prob_clusters' mode is selected the algorithm computes an agglomerative clustering over the probabilities
        of the iteration it of the mode. 
            In this case the use can select the dist_threshold parameter. If this param is 0.05,
            the dataset is divided in no more than 20 clusters; if it is 0.01, the datatset is divided 
            in maximum 50 clusters.
    - If 'ops_label' mode is selected, the algorithm computesp the conformity with respect to the nodel labels
        'optimistic' (agent probability <=0.33), 'pessimistic' (agent probability >0.66) 
         or 'neutral' (agent probability in (0.33, 0.66])
    - If 'both' mode is selected both conformity measures are computed
    """
    node_conformity_dict = calculate_conformity_scores(graph = Graph, system_status=system_status, prior_prob=params, it=it, 
                                                              mode = 'ops_label', dist_threshold= 0.01)

    # Check if the conformity scores were calculated successfully
    if not node_conformity_dict:
        current_app.logger.error(f"Error calculating conformity scores for sim_id {sim_id}")
        raise MetricsError("Error calculating conformity scores in ConformityScore().")
    
    current_app.logger.info("Conformity scores calculated successfully.")
    return jsonify({'success': True,
                'msg': f'Conformity scores calculated successfully (iteration={it}).',
                'node_conformity': node_conformity_dict,
                'simulation_id': sim_id}), 200

@app.route('/conformity-plot', methods=['POST'])
def conformity_plot():
    """
    Routes to generate the conformity plot.
    You can use this route to generate plots from the conformity score or save it as png.
    Form data from frontend containing graph data (nodes, edges, graph_type) and the simulation parameters and results:
    - 'it': iteration of simulation the 'ops_label' conformity plot is related to
    - 'node_conformity': dictionary with the node conformity results
    - 'simulation_id': unique identifier for the simulation
    - 'save': True if the user wants to save the plots, otherwise the function should return the plots without saving.
    """
    # generate plots
    data = request.get_json()
    if not data:
        raise ValidationError("Request body in conformity_plot() must be valid JSON.", field="body")
    # session_id = data['session_id']
    # graph = current_app.simulations.get_graph(session_id)
    # model = current_app.simulations.get_model(session_id)
    
    # Get sim_id from frontend request
    # sim_id = current_app.simulations.get_sim_id(session_id)
    sim_id = data['simulation_id']
    save = bool(int(data.get('save', 0)))  # flag variable to check if save plots (1) or send to frontend (0)
    node_conformity_dict = json.loads(data.get('node_conformity', {}))
    
    try:
        it = int(data.get('iteration', -1))  # Default to last (-1) if not specified
    except (TypeError, ValueError):
        raise ValidationError("iteration in ConformityScore() must be an integer.", field="iteration")

    format_plot_output = 'base64' if not save else 'png'  # Default format for plots, can be changed to 'png' if needed (static plots)
    
    current_app.logger.info('Generating conformity distribution plots...')
    if save:
        simulation_folder = current_app.config['RESULTS_FOLDER']
        sim_path = os.path.join(simulation_folder, sim_id)
        sim_plots_path = os.path.join(sim_path, 'plots')
        os.makedirs(sim_plots_path, exist_ok=True)
    if format_plot_output == 'png':  # Save plots in PNG format
        if not save:
            simulation_folder = current_app.config['RESULTS_FOLDER']
            sim_path = os.path.join(simulation_folder, sim_id)
            sim_plots_path = os.path.join(sim_path, 'plots')
            os.makedirs(sim_plots_path, exist_ok=True)
        plot_dict = save_conformity_plots(node_conformity_dict, it=it, 
                                          PLOT_FOLDER = sim_plots_path, format='png')
        
        if plot_dict is None or not plot_dict["conformity_plot_url"] or not plot_dict["conformity_ops_label_plot_url"]:
            raise OSError('An error occurred while saving conformity plots for current simulation.')

        conformity_plot_url = f'{sim_id}/{plot_dict["conformity_plot_url"]}' if node_conformity_dict['prob_clusters'] else ''
        conformity_opinion_label_plot_url = f'{sim_id}/{plot_dict["conformity_ops_label_plot_url"]}' if node_conformity_dict['ops_label'] else ''
        response = {
            'success': True,
            'msg': 'Conformity plot generated and saved successfully.',
            'conformity_plot_url': conformity_plot_url,
            'conformity_opinion_label_plot_url': conformity_opinion_label_plot_url,
            'simulation_id': sim_id
        }
        
    elif format_plot_output == 'base64':
        plot_dict = save_conformity_plots(node_conformity_dict, it=it, 
                                            PLOT_FOLDER = sim_plots_path, format='base_64')
        
        if plot_dict is None or not plot_dict["conformity_plot_url"] or not plot_dict["conformity_ops_label_plot_url"]:
            raise RuntimeError('An error occurred while generating conformity plots for current simulation.')
        response = {
            'success': True,
            'msg': 'Plots generated successfully.',
            'conformity_plot_url': plot_dict["conformity_plot_url"],
                'conformity_opinion_label_plot_url': plot_dict["conformity_ops_label_plot_url"],
            'simulation_id': sim_id
        }
      
    elif format_plot_output == 'svg':
        plot_dict = save_conformity_plots(node_conformity_dict, it=it, 
                                            PLOT_FOLDER = sim_plots_path, format='svg')
        if plot_dict is None or not plot_dict["conformity_plot_url"] or not plot_dict["conformity_ops_label_plot_url"]:
            raise RuntimeError('An error occurred while generating svg data for conformity plots for current simulation.')
        response = {
            'success': True,
            'msg': 'Conformity plot generated successfully.',
            'conformity_plot_svg': plot_dict["conformity_plot_svg"],
                'conformity_opinion_label_plot_svg': plot_dict["conformity_opinion_label_plot_svg"],
            'simulation_id': sim_id
        }
    else:
        raise ValidationError(f'Format plot type {format_plot_output} for conformity plot not valid. Use "png", "base_64" or "svg".')
    
    return jsonify(response)


if __name__ == '__main__':
    # Clean up old simulations before running new one
    # cleanup_old_simulations(days_old=7, simulation_folder=current_app.config['RESULTS_FOLDER'])
   app.run(debug=True)