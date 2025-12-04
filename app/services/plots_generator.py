import io
import base64
import os
import datetime
import exceptions.custom_exceptions
from almondo_model import OpinionDistribution, OpinionEvolution
from services.conformity_scores import plot_conformity_distribution, create_conformity_distribution_subplots
import matplotlib.pyplot as plt
from typing import Optional

def fig_to_base64(fig, transparent_bg=True):
    """Convert matplotlib figure to base64 string"""
    # Save to BytesIO buffer
    img = io.BytesIO()
    fig.savefig(img, format='png', dpi=300, 
                facecolor='none' if transparent_bg else 'white',
                bbox_inches='tight')
    img.seek(0)

    # Convert to base64
    plot_data = img.getvalue()
    img.close()
    plt.close(fig)  # Close the specific figure
    
    plot_url = base64.b64encode(plot_data).decode('utf-8')
    return f"data:image/png;base64,{plot_url}"

def fig_to_svg(fig, transparent_bg=True):
    """Convert matplotlib figure to SVG data"""
    # Save to StringIO buffer
    img = io.StringIO()
    fig.savefig(img, format='svg', dpi=300, 
                facecolor='none' if transparent_bg else 'white',
                bbox_inches='tight')
    img.seek(0)

    # Convert to base64
    svg_data = img.getvalue()
    img.close()
    plt.close(fig)  # Close the specific figure
    
    return svg_data

# Save final plots
def save_final_plots(system_status: dict, params: dict, PLOT_FOLDER: Optional[str], format: str = 'png') -> dict:
    """
    Save the final evolution and distribution plots to the specified directory.

    Arguments:
    - system_status: The dictionary containing the system status of the whole simulation.
    - params: The optimistic and pessimistic probabilities parameters used for the simulation. ('p_o' and 'p_p')
    - PLOT_FOLDER: The directory path where the plots will be saved.
    - format: The format in which to save the plots (default is 'png').
        - 'png': Save as PNG files.
        - 'base64': Return base64 encoded strings of the plots.
        - 'svg': Return SVG data of the plots.
    Returns:
    - A dictionary containing the URLs of the saved plots.
    """
    # if not hasattr(model, "system_status"):
    #    raise RuntimeError("You must run single_run() before calling save_plot()")
    
    # Ensure the plots folder exists
    os.makedirs(PLOT_FOLDER, exist_ok=True)
    # Generate unique filenames to avoid caching issues
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Generate and save the opinion distribution plot
    transparent_bg = True
    transparent_plot_area = True
    # Generate final opinion distribution plot
    actual_iteration = system_status[-1]['iteration']
    od = OpinionDistribution(system_status, params['p_o'], params['p_p'], values='probabilities')
    # Generate and save the opinion evolution plot
    oe = OpinionEvolution(system_status, params['p_o'], params['p_p'], kind='probabilities')
    
    if format == 'png':
        try:
            distribution_plot_filename = f"{timestamp}_final_probabilities_distribution_{actual_iteration}.png"
            distribution_plot_url = os.path.join(PLOT_FOLDER, distribution_plot_filename)
            opinion_evolution_plot_filename = f"{timestamp}_opinion_evolution_{actual_iteration}.png"
            opinion_plot_url = os.path.join(PLOT_FOLDER, opinion_evolution_plot_filename)
            od.plot(filename=distribution_plot_url, values='probabilities', stat=True, title=True,
                    figure_size=(10, 6), grid=True,
                    transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area, show=False)
            oe.plot(opinion_plot_url, figure_size=(10, 6), grid=True,
                    transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area, show=False)
        
            response = {
            'opinion_plot_url': f"/plots/{opinion_evolution_plot_filename}",
            'final_results_plot_url': f"/plots/{distribution_plot_filename}"
            }
        except Exception as e:
            raise OSError(f"Error saving plots of simulation results: {str(e)}")
    else:
        # generate figure for base64 or svg
        fig_od = od.plot(filename=None, values='probabilities', stat=True, title=True,
                         figure_size=(10, 6), grid=True,
                         transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area,show=False) 
        fig_oe = oe.plot(filename=None, figure_size=(10, 6), grid=True,
                         transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area,show=False)

    if format == 'base64':
        try:
            distribution_plot_url_base64 = fig_to_base64(fig_od, transparent_bg=True)
            opinion_plot_url_base64 = fig_to_base64(fig_oe, transparent_bg=True)
            response = {
                'opinion_plot_url': opinion_plot_url_base64,
                'final_results_plot_url': distribution_plot_url_base64
            }
        except Exception as e:
            raise RuntimeError(f"Error generating base64 plots of simulation results: {e}")
    elif format == 'svg':
        try:
            distribution_plot_svg_data = fig_to_svg(fig_od, transparent_bg=True)
            opinion_plot_svg_data = fig_to_svg(fig_oe, transparent_bg=True)
            response = {
                'opinion_plot_svg': opinion_plot_svg_data,
                'final_results_plot_svg': distribution_plot_svg_data
            }
        except Exception as e:
            raise RuntimeError(f"Error generating svg data for plotting simulation results: {e}")
    return response

# Save final plots
def save_conformity_plots(node_conformity_dict: dict, it: int, PLOT_FOLDER: Optional[str], format: str = 'png'):
    """
    Save the conformity plots to the specified directory or encoded it 
    to send to the frontend.

    Arguments:
    - node_conformity_dict: node conformity scores dictionary (see calculate_conformity_scores() function for its structure)
    - it: iteration of the simulation results the conformity scores are referred to.
    - PLOT_FOLDER: The directory path where the plots will be saved.
    - format: The format in which to save the plots (default is 'png').
        - 'png': Save as PNG files.
        - 'base64': Return base64 encoded strings of the plots.
        - 'svg': Return SVG data of the plots.
    Returns:
    - A dictionary containing the URLs of the saved plots.
    """
    if node_conformity_dict is None or (not node_conformity_dict['prob_clusters'] and not node_conformity_dict['ops_label']):
        raise RuntimeError("Empty node conformity scores dictionaries: you must run calculate_conformity_scores() before calling save_conformity_plot()")
    
   
    
    # timestamp = str(int(time.time()))
    # Generate and save the opinion distribution plot
    transparent_bg = True
    transparent_plot_area = True
    
    conformity_plot_url = True if node_conformity_dict['prob_clusters'] else False
    conformity_ops_label_plot_url = True if node_conformity_dict['ops_label'] else False

    if conformity_ops_label_plot_url: # node_conformity_dict['ops_label']:
        # Create dictionaries for ops label histograms subplot figure
        sub1_configs = [
            {'data': node_conformity_dict['ops_label']['optimistic']['node_conformity'], 
            'conformity': node_conformity_dict['ops_label']['optimistic']['global_conformity'], 
            'color': 'lightgreen', 'label': 'Optimistic'},
            {'data': node_conformity_dict['ops_label']['pessimistic']['node_conformity'], 
            'conformity': node_conformity_dict['ops_label']['pessimistic']['global_conformity'],  
            'color': 'lightsalmon', 'label': 'Pessimistic'}, 
            {'data': node_conformity_dict['ops_label']['neutral']['node_conformity'], 
            'conformity': node_conformity_dict['ops_label']['neutral']['global_conformity'],
            'color': 'lightblue', 'label': 'Neutral'}
        ]
        sub3_configs = [
            {'data': node_conformity_dict['ops_label']['overall']['node_conformity'], 
            'conformity': node_conformity_dict['ops_label']['overall']['global_conformity'],
            'color': 'lightgrey', 'label': 'Overall ops'}
        ]

    if format == 'png':
        # Ensure the plots folder exists
        os.makedirs(PLOT_FOLDER, exist_ok=True)
        # Generate unique filenames to avoid caching issues
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create the histogram visualization of the conformity scores distribution
        if conformity_plot_url: # check if the conformity score has been computed
            # Generate conformity scores distribution plot filename
            conformity_plot_filename = f"{timestamp}_conformity_distribution_{it}.png"
            conformity_plot_url = os.path.join(PLOT_FOLDER, conformity_plot_filename)
            # Create the histogram visualization of the conformity scores distribution
            try:
                plot_conformity_distribution(node_conformity_dict['prob_clusters']['node_conformity'], 
                                            node_conformity_dict['prob_clusters']['global_conformity'],
                                            filename = conformity_plot_url, ax= None,
                                            stat=True, title=True, color = 'lightblue', label='Clusters', legend=True, 
                                            figure_size=(10, 6), grid=False,
                                            transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area)
                plt.close()
            except Exception as e:
                raise OSError(f"Error saving conformity plot: {str(e)}")


        if conformity_ops_label_plot_url:# node_conformity_dict['ops_label'] # check if the conformity of opinion label has been computed
            # Generate conformity scores distribution plot filename
            conformity_ops_label_plot_filename = f"{timestamp}_ops_label_conformity_distribution_{it}.png"
            conformity_ops_label_plot_url = os.path.join(PLOT_FOLDER, conformity_ops_label_plot_filename)
            try:
                create_conformity_distribution_subplots(sub1_configs, sub3_configs,
                                  filename = conformity_ops_label_plot_url, ax = None, 
                                  stat=True, title=True, legend=True, figure_size=(10, 8), grid=False,
                                  transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area)
                plt.close()
            except Exception as e:
                raise OSError(f"Error saving conformity of opinion label plot: {str(e)}")
        
        response = {
        'conformity_plot_url': f"/plots/{conformity_plot_filename}" if conformity_plot_url else "",
        'conformity_ops_label_plot_url': f"/plots/{conformity_ops_label_plot_filename}" if conformity_ops_label_plot_url else ""
        }
    else:
        # generate figure for base64 or svg
        # Create the histogram visualization of the conformity scores distribution
        if conformity_plot_url: # check if the conformity score has been computed
            # Create the histogram visualization of the conformity scores distribution
            try:
                fig_c = plot_conformity_distribution(node_conformity_dict['prob_clusters']['node_conformity'], 
                                            node_conformity_dict['prob_clusters']['global_conformity'],
                                            filename = None, ax = None,
                                            stat=True, title=True, color = 'lightblue', label='Clusters', legend=True, 
                                            figure_size=(10, 6), grid=False,
                                            transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area)
            except Exception as e:
                raise RuntimeError(f"Error generating histogram for conformity scores: {e}")

        if conformity_ops_label_plot_url: # check if the conformity of opinion label has been computed
            try:
                fig_c_ol = create_conformity_distribution_subplots(sub1_configs, sub3_configs,
                                  filename = None, ax = None, 
                                  stat=True, title=True, legend=True, figure_size=(10, 8), grid=False,
                                  transparent_bg=transparent_bg, transparent_plot_area=transparent_plot_area)
            except Exception as e:
                raise RuntimeError(f"Error generating histogram for conformity of opinion label: {e}")

    if format == 'base64':
        try:
            if conformity_plot_url: 
                conformity_plot_url_base64 = fig_to_base64(fig_c, transparent_bg=True)
            if conformity_ops_label_plot_url: 
                conformity_opinion_label_plot_url_base64 = fig_to_base64(fig_c_ol, transparent_bg=True)
            response = {
                'conformity_plot_url': conformity_plot_url_base64 if conformity_plot_url else "",
                'conformity_opinion_label_plot_url': conformity_opinion_label_plot_url_base64 if conformity_ops_label_plot_url else ""
            }
        except Exception as e:
            raise RuntimeError(f"Error generating base64 plots of conformity scores: {e}")
    elif format == 'svg':
        try:
            if conformity_plot_url:
                conformity_plot_svg_data = fig_to_svg(fig_c, transparent_bg=True)
            if conformity_ops_label_plot_url:
                conformity_opinion_label_plot_svg_data = fig_to_svg(fig_c_ol, transparent_bg=True)
            response = {
                'conformity_plot_svg': conformity_plot_svg_data if conformity_plot_url else "",
                'conformity_opinion_label_plot_svg': conformity_opinion_label_plot_svg_data if conformity_ops_label_plot_url else ""
            }
        except Exception as e:
            raise RuntimeError(f"Error generating SVG plots of conformity scores: {e}")
    return response