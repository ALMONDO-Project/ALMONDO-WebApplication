# Almondo Simulator tool

This Simulator is a user-friendly web-application without User Interface designed to facilitate the simulation of the diffusion of influence about a topic across a network, including the influence of external lobbyists having different strategies. The agents of the network can be equipped with behavioural bias (confirmation bias) to a more realistic simulation.  

The simulation is based on the continutous agent-based opinion dynamics model named `AlmondoModel`, which extends the `DiffusionModel` class from the `NDlib` Pyhton library, and enables the simulation of opinion evolution and the effects of lobbying activities over time. The model can be customized with various parameters such as probabilities for optimistic and pessimistic events, node influence factors, and more.

The simulator is developed with the Python Flask framework and provides an effective backend for users to create agent networks, configure model parameters, run simulations, and visualize results effectively. Whether you're a researcher, analyst, or decision-maker, the web app simplifies the process of running complex simulations and extracting meaningful insights.

## Key Features:

- **Agent Network Creation:** Define and configure relationships between agents in the simulation using a variety of graphes;
- **Model Parameter Configuration:** Set initial conditions of the model and customize model parameters with flexible options, including event probabilities, and behavioural bias of agents;
- **Lobbyists modeling:** Lobbyists can have either optimistic or pessimistic models influencing the diffusion process and can be equipped with a custom startegy.
- **Simulation Execution:**  Run simulations based on predefined settings and input parameters, save results and continue simulation;
- **Steady-state detection**: The model can run until it reaches a steady state or the maximum iteration limit is reached.
- **Result Visualization:**  Generate graphical representation of the simulation results (evolution plot, opinion distribution, ...) for better interpretation of outcomes and save them.
- **Data Analysis:** Compute metrics and statistics both of the graph (overall and node-based), and simulation results (conformity score, opinion statistics, ...)

## Usage via Docker [Recommended]
🐳 Run the ALMONDO application with Docker Compose

To quickly start both the backend and frontend services using Docker Compose:
```bash
docker compose up --build
```

This command will:

- Start all containers defined in docker-compose.yml,
- Expose the web application on http://localhost:3000 (frontend),
- Run the backend API on http://localhost:8000.

To run in detached mode (in the background):
```bash
docker compose up -d
```

To stop and remove containers:
```bash
docker compose down
```

## Local Installastion

Requirement: Python 3.x

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/ALMONDO-Project/ALMONDO-WebApplication.git
cd app
pip install -r requirements.txt
```
### Installation of two required packages
Execute one of the following commands to install the `AlmondoModel` and `conformity` packages, if they are not successfully installed from the previous step:

```bash
# Direct from Git
pip install https://github.com/ALMONDO-Project/ALMONDO-Model.git
pip install https://github.com/GiulioRossetti/conformity.git

# Or in requirements.txt
git+https://github.com/ALMONDO-Project/ALMONDO-Model.git#egg=almondo-model
git+https://github.com/GiulioRossetti/conformity.git#egg=node_conformity

# Development mode
pip install -e git+https://github.com/ALMONDO-Project/ALMONDO-Model.git#egg=almondo-model
pip install -e git+https://github.com/GiulioRossetti/conformity.git#egg=node_conformity
```
**NOTE:**If the conformity package has any installation issue, use the following repo.  
```bash
# Direct from Git
pip install https://github.com/verdiana01/conformity.git

# Or in requirements.txt
git+https://github.com/verdiana01/conformity.git#egg=node_conformity

# Development mode
pip install -e git+https://github.com/verdiana01/conformity.git#egg=node_conformity
```
### (Optional) Installation of the User Interface
An example of web Interface the user can link to the project to directly launch simulation in a user-friendly way is the following:
```bash
git clone https://github.com/ALMONDO-Project/ALMONDO-WebInterface.git
cd almondo-web-app
```
Then follow the instruction reported into the `README.md` file to run the web interface. 

## Usage of the app
The ALMONDO-WebApplication can be used to simulate influence diffusion on a network with lobbyist interventions. Below is an example usage:
- If required, configure the base directory of the simulations. Default option is the `data` folder at the same level of `app` folder. To configure the base directory modifying the following line of the `config.py` file.
```bash
# Base paths
    DATA_BASE_URL = '../data'
```
- Run the application:
```bash
cd app
python app.py
```
**NOTE:** If you are developing new features locally, remember to add these lines of code to the file `app.py`:
```bash
if __name__ == '__main__':
   app.run(debug=True)
```
The application information of simulation states and errors both on the terminal and on a `error.log` file.
In debug mode the information are stored on a `debug.log` file.

## User Workflow
### Step 1: Creating the Agent Network
User can create the graph if the following way:
1. Choose the type of graph you want to use. Options available are:  'erdos_renyi', 'watts_strogatz', 'barabasi_albert', 'complete_graph', 'edgelist' or 'adjacency_matrix'.
2. Adjust its parameters as needed.
After creating the graph, it is automatically saved on the directory `BASE_DIRECTORY/uploads/graph/generated_graphs/`.

**NOTE**: For uploading a custom graph from a file ('edgelist' and 'adjacency matrix' options): 
- Use the 'edgelist' option to upload a .edglist or .txt file
- Use 'adjacency matrix' option to upload a .csv file containing the adjacency matrix of a graph. 
In the directory `data/uploads/graphs/user_graphs/` are reported two examples of custom graph files. 


### Step 2: Setting Model Parameters and Initial Conditions 
User can set model paramters and initialize model conditions (initial weights in [0; 1] of the agents in the network) using different methods, providing flexibility for a wide range of simulation scenarios. 

1. Define the initial conditions of the model and related parameters. Available options are: 
- 'uniform': uniform distribution on a specified internal,
- 'unbiased': same value for all agents,
- 'gaussian_mixture': overlapping of two o more gaussian distributions, 
- 'user_defined': custom vector defined by the user, uploading a .txt or a .csv file.
For more instructions, check for Notes about initial status parameter.txt file in the repository.

   **NOTE:** For user defined initial statuses:

   a) The gaussian mixture initial status can only be uploaded in a .txt file with the following format:
   
      means: 0.6, 0.3, 0.1
      stds: 0.44, 0.34, 0.22
      weights: 0.92, 0.01, 0.07

   The sum of "weights" must be 1.
Each element of the lists corresponds to the parameters of each gaussian distribution. In this example the final initial conditions will generated by the overlapping of three gaussian distributions.       

   b) The user defined initial status can only be uploaded in a .txt or .csv format:

   - in the .txt format, write a single column with the list of the initial status for each node 

   - in the .csv format, write a single column with the header Initial Status and the list of the initial status for each node. It is important to include the column header.

   Examples of both gaussian and 'user_defined' initial status files are included into the `data/uploads/initial_status/` folder.

2. Set the optimistic and pessimistic model probabilities $p_o$ and $p_p$. They represent how pessimistic or optimistic the agents are.

3. Choose the beahvioural biases for the agents: set the under-reaction bias $\lambda$ and directional motivated reasoning parameter $\phi$. User can provide them as a single value or a list.

### Step 3: Setting Lobbyists Parameters
User can run a simulation with or without the lobbyists. If you want to add lobbyists for each one yon must provide:
- The model: 'pessimistic' (0) or 'optimistic' (1)
- the strategy: a TxN matrix  of 0s and 1s where T is the number of steps the lobbyist is active during simulation and N is the number of agents in the network. If the lobbyist sends a signal to the agent j at iteration i, then the (i,j) position of the strategy matrix is 1; otherwise 0.

The strategy for each can be provide manually as a .txt or .csv file or automatically generated by the program based on the budget 'B', the active time steps 'T' and the cost 'c' of a signal.

After creating the startegies for each lobbyist, they are automatically saved on the specific simulation directory `BASE_DIRECTORY/simulation_results/{sim_id}/strategies/`.

### Step 4: Running the simulation
Once the agent network, the model parameters and its initial conditions as well as the lobbyists strategies are set, user can run the simulation based on their predefined settings and input.
At the beginning of the simulation the application create a unique simulation identifier and the related directory into the `simulation_results` folder.

At the end of the simulation the graph, the simulation configuration and final results are automatically saved into specific files into the simulation folder directory (`BASE_DIRECTORY/simulation_results/{sim_id}/`). 

### Step 5: Visualizing Results
The simulation outputs can be visualised for easy interpretation throughout the evolution plot of the agents'opinion over time and the opinion distribution histogram of the final iteration (or a specific one). 

The simulation plots can be saved in .png format in the simulation directory, under the `plot` folder, otherwise svg data o base64 encoding figure can be obtained.

### Step 6: Compute Graph and Simulation Metrics
User can compute the graph metrics or that one related to the simulation results. 

Examples of possible graph and node metrics are: degree, density, node degree, and bretweeness centrality.
Examples of simulation results metrics over agents' opinion in the final or a specific iteration are: mean, standard deviation, effective number of clusters, absolute paiwise distance of opinions, lobbyist performance, and conformity scores and plots.

The conformity plot can be saved in .png format in the simulation directory, under the `plot` folder, otherwise svg data o base64 encoding figure can be obtained.

### Files and Directories
- data/: Base directory for saving files and simulation results.
- data/simulation_results/:  Base directory for saving simulation results of all simulations. 
- data/simulation_results/{date_simid}/:  Directory where saving simulation results of a single simulation. 
- data/simulation_results/{date_simid}/config.json: The configuration file of the simulation.
- data/simulation_results/{date_simid}/status.json: File containing the data of the simulation.
- data/simulation_results/{date_simid}/strategies/: Directory where lobbyists' strategies are stored.
- data/simulation_results/{date_simid}/plots/: Directory where default figures are stored.
- data/simulation_results/{date_simid}/graph.csv: File containing the graph of the network.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
The project requires [NDlib](https://github.com/GiulioRossetti/ndlib), distributed under the BSD 2-Clause "Simplified" License. All modifications and original contributions are provided under MIT.
 
## Acknowledgements and funding declaration
This simulator was created as part of ongoing research into opinion dynamics and lobbying influence. It relies on NDlib for network-based modeling and networkx for graph management. See references for project website.
 
This study received funding from the European Union - Next-GenerationEU -National Recovery and Resilience Plan (NRRP) – MISSION 4 COMPONENT 2, INVESTMENT N. 1.1, CALL PRIN 2022 PNRR D.D. 1409 14-09-2022 – ALMONDO Project (Analyzing climate Lobbying with a simulation Model based ON Dynamic Opinions), CUP N. J53D23015400001.
Coordinator: Prof. Daniele Giachini, School of Advanced Studied  Sant'Anna, Pisa (PI), Italy
 
## References
- Giachini, D., Del Rosso, V., Ciambezi, L., Fornari, F., Pansanella, V., Popoyan, L., & Sîrbu, A. (2025). "Navigating the Lobbying Landscape: Insights from Opinion Dynamics Models." *arXiv preprint arXiv:2507.13767*.
- ALMONDO project website: https://almondo-project.github.io/
