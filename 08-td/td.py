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
def sarsa(g, epsilon_function, N = 10, gamma = 0.9, alpha=0.1, max_game_length = 100):
    # N = number of episodes
    # max_game_length = stops game after so many moves. Helps prevent bad policies slowing down the programme.

    # Initialise Q (arbitrarily)
    Q = {}
    for s, possible_actions in g.actions.items():
        for a in possible_actions:
            Q[(s, a)] = 0.0

    # Set terminal Q = 0
    terminal_states = set(g.all_states(include_terminal=True)) - set(g.all_states(include_terminal=False))
    for s in terminal_states:
        Q[(s, None)] = 0.0

    for n in range(1, (N+1)):
        g.reset()
        s1 = g.current_state()
        a1 = epsilon_greedy_action(g=g, Q=Q, s=s1, epsilon=epsilon_function(n))
        
        game_over = False
        i = 0
        while (not game_over) and (i < max_game_length):
            i += 1
            r = g.move(a1)
            s2 = g.current_state()

            game_over = g.game_over()
            if game_over:
                a2 = None
            else:
                a2 = epsilon_greedy_action(g=g, Q=Q, s=s2, epsilon=epsilon_function(n))
            
            #print((s1, a1, r, s2, a2))

            # Update
            Q[(s1, a1)] = Q[(s1, a1)] + alpha*(r + gamma * Q[(s2, a2)] - Q[(s1, a1)])
            s1 = s2
            a1 = a2 
        
    # Get policy from Q function
    policy, value_f = get_policy_and_value_function(Q, g)
    
    return(policy, value_f)

## SARSA helpers
def epsilon_greedy_action(g, Q, s, epsilon):
    p = np.random.sample()
    if p < epsilon:
        # random action
        action = np.random.choice(g.actions[s])
    else:
        # greedy action
        for a in g.actions[s]:
            value = -np.Inf
            if Q[(s, a)] > value:
                value = value
                action = a
    return(action)

def get_policy_and_value_function(Q, g):
    policy = {}
    value_f = {}
    for s in g.all_states(include_terminal=False):
        value = -np.Inf
        for a in g.actions[s]:
            if Q[(s, a)] > value:
                value = Q[(s, a)]
                action = a
        policy[s] = action
        value_f[s] = value
    return(policy, value_f)

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
        out[key[0], key[1]] = value
    print(out)
