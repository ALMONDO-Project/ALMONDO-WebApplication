import os
import numpy as np
import random

def create_single_random_strategy(B: int, # total budget
                                    T: int, # total time steps
                                    N: int, # number of agents
                                    c: int =1 # cost of a signal
                                    ) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Create the strategy matrix TxN and randomly selects B/c signals in the TxN matrix to set equals to 1.

    Args:
        B (int): The total budget of lobbyist
        T (int): The number of active time steps of lobbyist
        N (int): The number of agents
        c (int): The cost to send a signal

    Returns:
        numpy.ndarray: A matrix TxN of 0s with B/c randomly selected elements set to 1.
        list: A list of the (row, column) indices that were set to 1, i.e. the list of (time_step, agent) of sent signals
    """
    matrix = np.zeros((T, N), dtype=int)
    total_elements = T * N
    num_signals = B//c  # number of signals
    if num_signals > total_elements:
        print("Number of signals is greater than the total number of elements in the matrix."
              "Lobbyist will always send signals to all agents at each iteration.")
        num_signals = total_elements
        

    # Generate k unique random linear indices
    linear_indices = np.random.choice(total_elements, size=num_signals, replace=False)

    # Convert linear indices to row and column indices
    row_indices, col_indices = np.unravel_index(linear_indices, (T, N))

    # Create a list of (row, column) index pairs
    selected_indices = list(zip(row_indices, col_indices))

    # Set the corresponding elements in the matrix to 1
    matrix[row_indices, col_indices] = 1

    return matrix

def create_single_random_strategy_per_time(B: int, # total budget
                                             T: int, # total time steps
                                             N: int, # number of agents
                                             c: int =1 # cost of a signal
                                             ) -> np.ndarray:
    """
    Create the strategy matrix TxN, randomly selects fixed number of signals at each time step in the TxN matrix
      and sets them equals to 1. Per time step, the number of signals is fixed B/(c*T).

    Args:
        B (int): The total budget of lobbyist
        T (int): The number of active time steps of lobbyist
        N (int): The number of agents
        c (int): The cost to send a signal

    Returns:
        numpy.ndarray: A matrix TxN of 0s with randomly selected elements set to 1. Per time step, the number of signals is fixed B/(c*T).
        list: A list of the (row, column) indices that were set to 1, i.e. the list of (time_step, agent) of sent signals
    """
    inter_per_time = B // (c * T)
    matrix = np.zeros((T, N), dtype=int)
    for t in range(T):
        indices = np.random.choice(N, inter_per_time, replace=False)
        matrix[t, indices] = 1
    return matrix

def create_strategies(lobbyists_data: dict, sim_path: str, n_lobbyists: int) -> None:
    """
    Generate and save strategies for the lobbyists.
    Args:
        lobbyists_data (list): List of dictionaries containing lobbyist data.
            Each dictionary should contain 
            - 'm': model, 'B': budget, 'c': cost, 'T': active time steps.
        sim_path (str): Path to the simulation directory where strategies will be saved.
        n_lobbyists (int): Number of lobbyists for which strategies are to be created.
    """
   # print('Creating lobbyists strategies')  # TODO: Implement logging and remove print statements (put it outside the function)

    for id in range(n_lobbyists):
        data = lobbyists_data[id]
        B = data['B']
        c = data['c']
        T = data['T']
        folder = os.path.join(sim_path, 'strategies', str(B))
        os.makedirs(folder, exist_ok=True)
        # for run in range(self.nruns):
        filename = f'strategy_{id}.txt'
        path = os.path.join(folder, filename)
        if not os.path.exists(path):
            matrix = create_single_random_strategy(B, T, c)
            print('Saving strategy to file')
            np.savetxt(path, matrix, fmt="%i")
        else:
            continue
    #print('Strategies created')  # TODO: Implement logging and remove print statements (put it outside the function)

def read_random_strategy(B: int, strategies_path: str) -> tuple:
    """
    Read a random strategy for a given B value.

    Arguments:
    - B: The B value associated with the lobbyist's strategy.
    - strategies_path: The path to the directory containing the strategies.

    Returns:
    - tuple: A tuple containing the strategy matrix and its filename.
    """
    path = os.path.join(strategies_path, str(B))
    strategy_name = random.choice(os.listdir(path))
    filepath = os.path.join(path, strategy_name)
    return np.loadtxt(filepath).astype(int), filepath