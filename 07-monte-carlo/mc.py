# Glenn Thomas
# 2020-05-03
# Monte Carlo functions for reinforcement learning

import numpy as np
import pdb

# Prints value function
def print_value_function(V, g):
    out = np.zeros((g.height, g.width))
    for key, value in V.items():
        out[key[0], key[1]] = value
    print(np.round(out, 2))
    return(out)

# Print deterministic policy
def print_determinisitic_policy(policy, g):
    out = np.full(shape = (g.height, g.width), fill_value = " ")
    for key, value in policy.items():
        action = g.actions_array[value == value.max()][0]
        out[key[0], key[1]] = action
    print(out)

def get_value(g, policy, N = 100):
    # Gets the value function using monte carlo (using simulation)
    
    possible_starting_states = g.all_states(include_terminal=False)

    all_returns = {}
    for _ in range(N):
        g.reset()

        # Set random start
        starting_state_ind = np.random.choice(len(possible_starting_states))
        g.set_state(possible_starting_states[starting_state_ind])

        states, returns = g.play_game(policy)
        # Use first visit MC
        seen_states = set()
        for i in range(len(states)):
            s = states[i]
            if s not in seen_states:
                seen_states.add(s)
                # create state if not yet seen
                if s not in all_returns.keys():
                    all_returns[s] = [returns[i]]
                else:
                    all_returns[s].append(returns[i])
        
    for s, s_return in all_returns.items():
        all_returns[s] = np.array(s_return).mean()

    return(all_returns)

