# Glenn Thomas
# 2020-06-01
# TD (Temporal differcing functions)

import numpy as np
import pdb
import dynamic_programming_functions as dp

def create_epsilon_soft_policy(policy, g, eps = 0.01):
    # Implements epsilon soft on initial policy
    new_policy = policy.copy()
    for key, value in policy.items():
        # Create random component
        probs = np.zeros(len(g.actions_array))
        probs[np.isin(g.actions_array, g.actions[key])] = 1/len(g.actions[key])
        # Take random policy with probability eps
        new_policy[key] = eps*probs + (1.0-eps)*value

    return(new_policy)

def td0(g, policy, N = 10, gamma = 0.9, alpha = 0.1, epsilon = 0.01):

    # Use epsilon-soft
    policy = create_epsilon_soft_policy(policy, g, eps = epsilon)

    # Intialise V
    V = {}
    all_states = g.all_states(include_terminal=True)
    for s in all_states:
        V[s] = 0
    

    for _ in range(1, N + 1):
        # play game
        g.reset()
        state_log, state_action_log, G, reward_log = g.play_game(policy, gamma = gamma)
        # print(state_log)
        # print(reward_log)

        for t in range(len(state_log) - 1):
            s = state_log[t]
            s2 = state_log[t+1]
            r = reward_log[t+1]
            V[s] = V[s] + alpha*(r + gamma*V[s2] - V[s])
            # if (s == (2,0)) and (( _ % 100) == 0):
            #     print(np.round(V[s], 2))

        # if _ % 100 == 0:
        #     print("")
        #     dp.print_value_function(V, g)

    return(V)

## SARSA
def sarsa(g, epsilon_function, N = 10, gamma = 0.9, alpha=0.1):
    
    # Initialise Q
    Q = {}
    for s, possible_actions in g.actions.items():
        for a in possible_actions:
            Q[(s, a)] = 0.0

    # Set terminal Q = 0
    terminal_states = set(g.all_states(include_terminal=True)) - set(g.all_states(include_terminal=False))
    for s in terminal_states:
        Q[(s, None)] = 0.0

    print(Q)

    # For t = 1..N