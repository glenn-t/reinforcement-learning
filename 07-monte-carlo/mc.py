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

def get_value(g, policy, N = 100, gamma = 0.9):
    # Gets the value function using monte carlo (using simulation)
    # (Policy evaluation)
    
    possible_starting_states = g.all_states(include_terminal=False)

    all_returns = {}
    for _ in range(N):
        g.reset()

        # Set random start
        starting_state_ind = np.random.choice(len(possible_starting_states))
        g.set_state(possible_starting_states[starting_state_ind])

        states, returns = g.play_game(policy, gamma = gamma)
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

def mc_policy_improvement(g, gamma = 0.9, N = 1000):
    # Does policy improvement using monte carlo
    # Uses the Q function instead of V
    # Does not reset Q after each policy evaluation.
    # Policy updates after each episode
    # Uses random starts and action method.

    possible_starting_states = g.all_states(include_terminal=False)
    init_policy = np.zeros(len(g.actions_array))

    # Logs samples of Q
    all_returns = {}
    Q = {} # technically this should be initialised arbitrarily, but I workaround that.

    # initialise random policy
    random_policy = g.actions.copy()
    for key, value in random_policy.items():
        probs = np.zeros(len(g.actions_array))
        probs[np.isin(g.actions_array, value)] = 1/len(value)
        random_policy[key] = probs

    policy = random_policy.copy()

    for n_iter in range(1, N + 1):
        # Set random start and action
        starting_state_ind = np.random.choice(len(possible_starting_states))
        starting_state = possible_starting_states[starting_state_ind]
        init_policy = random_policy[starting_state]

        # play game
        state_action_log, returns = g.play_game(policy, gamma = gamma, init_policy = init_policy, init_state = starting_state, return_actions = True)



        ## Add state_action_log to dataset and update Q
        # Use first visit MC
        seen_states = set()
        for i in range(len(state_action_log)):
            s_a = state_action_log[i]
            if s_a not in seen_states:
                seen_states.add(s_a)
                # if state already seen in any episode, then just append to data
                if s_a in all_returns:
                    all_returns[s_a].append(returns[i])
                else:
                    # if state not seen in any episode so far, create it
                    all_returns[s_a] = [returns[i]]

                # Update Q
                Q[s_a] = np.mean(all_returns[s_a])

        ## Update policy
        # TODO - this part could be optimised to remove the double for loop
        for s in policy:
            # Find all entries in Q with state=s
            new_action = None
            max_value = -np.Inf
            for key, value in Q.items():
                if key[0] == s and value > max_value:
                    new_action = key[1]
                    max_value = value

            if new_action is not None:
                policy[s] = np.zeros(len(g.actions_array))
                policy[s][g.actions_array == new_action] = 1

        # Could track the value function of the starting position
        # print value function of starting position, for every 100th itera
        if (n_iter % 100) == 0:
            action = g.actions_array[np.where(policy[g.start])[0]][0]
            print(Q[(g.start, action)])

    # Get value function
    V = {}
    for s in policy:
        # will fail if any still has determinisitic value function
        action = g.actions_array[np.where(policy[s] == 1.0)][0]
        V[s] = Q[(s,action)]


    return((V, policy))