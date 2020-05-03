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
                # if state already seen in any episode, then just append to data
                if s in all_returns:
                    all_returns[s].append(returns[i])
                else:
                    # if state not seen in any episode so far, create it
                    all_returns[s] = [returns[i]]

    ## Note - this could be made more memory efficient by not
    # storing all_returns, but updating the sample mean each time.
    for s, s_return in all_returns.items():
        all_returns[s] = np.array(s_return).mean()

    return(all_returns)

