import ndlib.models.ModelConfig as mc
from almondo_model import AlmondoModel, ALMONDOSimulator, OpinionDistribution, OpinionEvolution
from services.metrics_and_statistics import *
from services.conformity_scores import probabilities_clustering, compute_conformity_scores, plot_conformity_distribution, compute_conformity_scores_opinion
from almondo_model.functions.utils import transform
from services.file_manager import save_conformity_plots
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import seaborn as sns
import os
import json
from tqdm import tqdm
import datetime

# Simulation parameters
N = 1000 # number of agents in the simulation
nlob = 2 # number of lobbyists in the simulation
int_rate = 0.2 # interaction reate of the lobbyists per time-step
T = 3000 # max number of active time steps of lobbyists
b = int(int_rate*N*T) # budget of lobbyists in the simulation
seed = None 
G = nx.erdos_renyi_graph(N, p=0.05, seed=seed)  # Example graph
# read the graph from file
# G = nx.read_edgelist(file_path+edgelist_file, delimiter=",")
its = 400 # number of iterations for the simulation
sim_option = 'bunch'  # options: 'bunch' for iteration bunch or 'ss' for steady state
# Other simulation parameters
params = {
    'N': N,
    'n_lobbyists': nlob,
    'p_o': 0.01,
    'p_p': 0.99,
    'initial_distribution': 'uniform',
    'T': 10000, # max number of iterations
    'lambda_values': [1.0],
    'phi_values': [0.05],
    'base': '../results'
}
params['nruns'] = 1 # number of runs with the same parameters

params['scenario'] = f'{N}_agents/{nlob}_lobbyists/'

if nlob > 0:
    params['lobbyists_data'] = dict()
    for id in range(nlob):
        params['lobbyists_data'][id] = {'m': id%2, 'B': b, 'c': 1, 'strategies': [], 'T': T}

os.makedirs(params['base'], exist_ok=True)
path = os.path.join(params['base'], params['scenario'])
os.makedirs(path, exist_ok=True)

with open(os.path.join(path, 'initial_config.json'), 'w') as f:
    json.dump(params, f, indent=4)


simulator = ALMONDOSimulator(**params, verbose=False)
print(f'Starting configuration lambda={params["lambda_values"][0]}, phi={params["phi_values"][0]}')
simulator.config_path = os.path.join(simulator.scenario_path, f'{params["lambda_values"][0]}_{params["phi_values"][0]}')
os.makedirs(simulator.config_path, exist_ok=True)

# Model configuration

print('Creating configuration object')
config = mc.Configuration()

print('Assigning p_o and p_p parameters')
config.add_model_parameter("p_o", simulator.p_o)
config.add_model_parameter("p_p", simulator.p_p)
print(f'p_o={simulator.p_o}, p_p={simulator.p_p}')

# Configure lambda values for each agent
if isinstance(simulator.lambdas[0], list):
    for i in simulator.graph.nodes():
        config.add_node_configuration("lambda", i, simulator.lambdas[0][i])
elif isinstance(simulator.lambdas[0], float):
    print('Assigning homogeneous lambda')
    for i in simulator.graph.nodes():
        config.add_node_configuration("lambda", i, simulator.lambdas[0])
else:
    raise ValueError("lambda_v must be a float or a list")

# Configure phi values for each agent
if isinstance(simulator.phis[0], list):
    for i in simulator.graph.nodes():
        config.add_node_configuration("phi", i, simulator.phis[0][i])
elif isinstance(simulator.phis[0], float):
    print('Assigning homogeneous phi')
    for i in simulator.graph.nodes():
        config.add_node_configuration("phi", i, simulator.phis[0])
else:
    raise ValueError("phi_v must be a float or a list")

# Initialize the model with the graph and configuration
print('Configuring model: assigning graph, parameters, and initial distribution of weights')
simulator.graph = G
simulator.model = AlmondoModel(simulator.graph, seed=seed)
simulator.model.set_initial_status(config, kind= params['initial_distribution'], status=[])

print('Assign strategies to lobbyists if any')
if simulator.n_lobbyists > 0:
    for id in tqdm(simulator.lobbyists_data):
        data = simulator.lobbyists_data[id]
        B = data['B']
        m = data['m']
        matrix, name = simulator.read_random_strategy(B)
        # Add lobbyist with strategy to the model
        simulator.model.add_lobbyist(m, matrix)
        simulator.lobbyists_data[id]['strategies'].append(name)

print('Configuration ended')

# Simulation loop
for run in range(params['nruns']):
    run_dir = os.path.join(simulator.config_path, f'run{run}')
    os.makedirs(run_dir, exist_ok=True)

    plot_dir = os.path.join(run_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    nx.write_edgelist(simulator.graph, f"{run_dir}/graph.edgelist", delimiter=',')


    # Run iterations
    if sim_option == 'bunch':
        iterations = simulator.model.iteration_bunch(T=its)
        simulator.system_status = iterations

        # Calculate the final weights and probabilities
        fws = [el for el in simulator.system_status[-1]['status'].values()]
        fps = transform(fws, simulator.p_o, simulator.p_p)  # optimistic model

        fd = {
            'final_weights': fws,
            'final_probabilities': fps,
            'final_iterations': int(simulator.system_status[-1]['iteration'])
        }

    elif sim_option == 'ss': # steady state simulation
        steady_state = simulator.model.steady_state(drop_evolution=True)
        simulator.system_status = steady_state
    
    # Save system status and configuration to a file
    simulator.save_system_status(run_dir)
    simulator.save_config()
  
    # Save figures
    print('Saving final plots and configuration...')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    distribution_plot_filename = f"{timestamp}_final_probabilities_distribution_{simulator.model.actual_iteration}.png"
    distribution_plot_url = os.path.join(plot_dir, distribution_plot_filename)
    transparent_bg = False
    transparent_plot_area = False
    od = OpinionDistribution(simulator.model.system_status, simulator.p_o, simulator.p_p, values='probabilities')
    # Generate and save the opinion evolution plot
    opinion_evolution_plot_filename = f"{timestamp}_opinion_evolution_{simulator.model.actual_iteration}.png"
    opinion_plot_url = os.path.join(plot_dir, opinion_evolution_plot_filename)
    oe = OpinionEvolution(simulator.model.system_status, simulator.p_o, simulator.p_p, kind='probabilities')

    od.plot(filename=distribution_plot_url, values='probabilities', stat=True, title=True,
            figure_size=(10, 6), grid=True,
            transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area)
    oe.plot(opinion_plot_url, figure_size=(10, 6), grid=True, 
            transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area)

    # Specific parameters where compute metrics
    print('Computing statistics and metrics...')
    node_id = 10
    it = -1
    graph_metrics, error_basic = graph_basic_metrics(graph=G)
    node_info, error_node = get_node_info(graph=G, model=simulator.model, node_id = 10, it=it,betweeness=False)
    opinion_statistics, error_stats = get_opinion_statistics(model=simulator.model, it=it)
    opinion_metrics, error_metrics = calculate_opinion_metrics(model=simulator.model, it=it)

    

    # Plot the graph using matplotlib
    graph_plot_filename = f"{timestamp}_graph_plot_{simulator.model.actual_iteration}.png"
    graph_plot_url = os.path.join(plot_dir, graph_plot_filename)
    plt.figure(figsize=(8, 6))
    fig_nx = nx.draw(G,with_labels=False, node_color='gray', node_size=100, edge_color='gray')
    transparent_bg = False
    plt.title("Erdos-Renyi Graph Visualization")
    if graph_plot_filename is not None:
            bg_color = 'none' if transparent_bg else 'white'
            plt.savefig(graph_plot_url, dpi=300, facecolor=bg_color, bbox_inches='tight')
    else:
            plt.show()
    plt.close()

    # Read the CSV file of final weights
    # df = pd.read_csv(file_path+file_name) --> modificare per JSON
    # Open and read the JSON file of system status
    # with open('data.json', 'r') as file:
    #    system_status = json.load(file)
    
    # Calculate conformity scores
 
    node_conformity_dict, error = calculate_conformity_scores(graph = G, model = simulator.model, it=-1, mode = 'both', dist_threshold= 0.01)

    # node_conformity_dict = {'prob_clusters': {}, 'ops_label': {}}
    """"
    conformity_plot_filename = f"{timestamp}_conformity_plot_{simulator.model.actual_iteration}.png"

    # Create the histogram visualization of the conformity scores distribution
    fig = plot_conformity_distribution(node_conformity, global_conformity,
                                            filename = conformity_plot_filename,
                                            stat=True, title=True, 
                                            color = 'lightblue', label='Clusters', legend=True, 
                                            figure_size=(10, 6), grid=False,
                                            transparent_bg=False, transparent_plot_area=False)
    plt.close()
     # Print global conformity
    print(f"Global conformity score: {global_conformity:.4f}")
    

    fig_opt,ax_opt = plot_conformity_distribution(node_conformity_opt, global_conformity_opt,
                                        filename = None, # f"{timestamp}_conformity_plot_ops_opt_{simulator.model.actual_iteration}.png",
                                        stat=True, title=True, ax=None,
                                        color = 'lightgreen',  label='Optimistic', legend=True,
                                        figure_size=(10, 6), grid=False,
                                        transparent_bg=False, transparent_plot_area=False)
    fig_pess,ax_pess = plot_conformity_distribution(node_conformity_pess, global_conformity_pess,
                                    filename = None, #f"{timestamp}_conformity_plot_ops_pess_{simulator.model.actual_iteration}.png",
                                    stat=True, title=True, ax=ax_opt,
                                    color = 'lightsalmon',  label='pessimistics', legend=True,
                                    figure_size=(10, 6), grid=False,
                                    transparent_bg=False, transparent_plot_area=False)
    fig_neut, ax_neut = plot_conformity_distribution(node_conformity_neut, global_conformity_neut,
                                    filename = f"{timestamp}_conformity_plot_ops_{simulator.model.actual_iteration}.png",
                                    stat=True, title=True, ax=ax_pess,
                                    color = 'lightblue',  label='neutral', legend=True,
                                    figure_size=(10, 6), grid=False,
                                    transparent_bg=False, transparent_plot_area=False)
    plt.close()
    fig2, ax_over = plot_conformity_distribution(node_conformity_ops, global_conformity_ops,
                                        filename = f"{timestamp}_conformity_plot_ops_overall_{simulator.model.actual_iteration}.png",
                                        stat=True, title=True, ax=None,
                                        color = 'lightgrey',  label='ops', legend=True,
                                        figure_size=(10, 6), grid=False,
                                        transparent_bg=False, transparent_plot_area=False)
    plt.close()

    """
    # Put the last two plots in a single subplot
    
    """
    # Usage
    # q1_data = [data1, data2, data3]  # 3 datasets for Quarter 1
    # q3_data = single_dataset         # 1 dataset for Quarter 3
    # Quarter 1 - Three overlapped histograms
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
    plt.close()
    """
    if node_conformity_dict is None or not node_conformity_dict:
        response = {'success': False, 
                        'msg': 'Error in calculating conformity score: no output',
                        'error': 'Error in calculating conformity score: no output'}
        print(response)
        continue
    # Handle errors
    if error:
        response = {'success': False, 
                        'msg': 'Error in calculating conformity score',
                        'error': error}
        print(response)
        continue
    # generate plots
    format_plot_output = 'png'
    if format_plot_output == 'png':  # Save plots in PNG format
        try:
            plot_dict = save_conformity_plots(node_conformity_dict, it=simulator.model.system_status[it]['iteration'], 
                                              PLOT_FOLDER = plot_dir, format='png')
            if plot_dict is None or not plot_dict["conformity_plot_url"] or not plot_dict["conformity_ops_label_plot_url"]:
                print(f"Error generating plots for sim_id run{run}")
                response = {
                    'success': False,
                    'message': 'An error occurred while generating plots for current simulation.',
                    'error': 'An error occurred while generating plots for current simulation.'
                }
            else:
                conformity_plot_url = f'run{run}/{plot_dict["conformity_plot_url"]}' if node_conformity_dict['prob_clusters'] else ''
                conformity_opinion_label_plot_url = f'run{run}/{plot_dict["conformity_ops_label_plot_url"]}' if node_conformity_dict['ops_label'] else ''
                response = {
                    'success': True,
                    'conformity_plot_url': conformity_plot_url,
                    'conformity_opinion_label_plot_url': conformity_opinion_label_plot_url,
                    'simulation_id': f'run{run}'
                }
        except Exception as e:
            print(f"Error generating plots for sim_id run{run}: {e}")
            response = {
                'success': False,
                'message': 'An error occurred while generating plots for current simulation.',
                'error': str(e)
            }
        
    elif format_plot_output == 'base64':
        try:
            plot_dict = save_conformity_plots(node_conformity_dict, it=simulator.model.system_status[it]['iteration'], 
                                              PLOT_FOLDER = plot_dir, format='base_64')
            if not plot_dict["conformity_plot_url"] or not plot_dict["conformity_ops_label_plot_url"]:
                print(f"Error generating plots for sim_id run{run}")
                response = {
                    'success': False,
                    'message': 'An error occurred while generating plots for current simulation.',
                    'error': 'An error occurred while generating plots for current simulation.'
                }
            else:
                response = {
                    'success': True,
                    'conformity_plot_url': plot_dict["conformity_plot_url"],
                    'conformity_opinion_label_plot_url': plot_dict["conformity_ops_label_plot_url"],
                    'simulation_id': f'run{run}'
                }
        except Exception as e:
            print(f"Error generating plots for sim_id run{run}: {e}")
            response = {
                'success': False,
                'message': 'An error occurred while generating plots for current simulation.',
                'error': str(e)
            }
    elif format_plot_output == 'svg':
        try:
            plot_dict = save_conformity_plots(node_conformity_dict, it=simulator.model.system_status[it]['iteration'], 
                                              PLOT_FOLDER = plot_dir, format='svg')
            if not plot_dict["conformity_plot_url"] or not plot_dict["conformity_ops_label_plot_url"]:
                print(f"Error generating plots for sim_id run{run}")
                response = {
                    'success': False,
                    'message': 'An error occurred while generating plots for current simulation.',
                    'error': 'An error occurred while generating plots for current simulation.'
                }
            else:
                response = {
                    'success': True,
                    'conformity_plot_svg': plot_dict["conformity_plot_svg"],
                    'conformity_opinion_label_plot_svg': plot_dict["conformity_opinion_label_plot_svg"],
                    'simulation_id': f'run{run}'
                }
        except Exception as e:
            print(f"Error generating plots for sim_id run{run}: {e}")
            response = {
                'success': False,
                'message': 'An error occurred while generating plots for current simulation.',
                'error': str(e)
            }
    print(response)