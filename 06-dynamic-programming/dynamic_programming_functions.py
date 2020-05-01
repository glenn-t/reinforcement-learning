import numpy as np
import pdb

# interative_policy_evaluation
# Finds value function of a given policy
# Note, this may not converge if the policy creates loops that cannot reach
# the terminal state. So need to include max_iter

def get_value(policy, g, gamma = 0.9, tol = 1e-6, max_iter = 1000):
    V = {}
    for s in g.all_states(include_terminal=True):
        V[s] = 0 # terminal states have 0 as value

    not_small_enough = True
    iteration = 0
    while not_small_enough & (iteration < max_iter):
        delta = 0
        for state in V.keys():
            old_V = V[state]
            # V(s) only has value if it's not a terminal state
            if state in g.actions:
                # bellman equations
                sum = 0

                # generate probabilities
                p = policy[state]
                # get probability if random action is taken due to wind
                windy_prob = np.zeros(len(g.actions_array))
                windy_prob[np.isin(g.actions_array, g.actions[state])] = 1/len(g.actions[state])
                p = p*(1-g.windy) + windy_prob*g.windy

                # loop though all posible actions
                for i in range(len(policy[state])):
                    prob = p[i]
                    if (prob > 0):
                        g.set_state(state)
                        #pdb.set_trace()
                        reward = g.move(g.actions_array[i], force = True)
                        sum += prob*(reward + gamma*V[g.current_state()])
                V[state] = sum
                delta = max(delta, np.abs(V[state] - old_V))
        not_small_enough = delta > tol
        iteration += 1
    
    if(iteration == max_iter):
        print("Did not converge")
    return(V)

# Prints value function
def print_value_function(V, g):
    out = np.zeros((g.height, g.width))
    for key, value in V.items():
        out[key[0], key[1]] = value
    print(np.round(out, 2))
    return(out)

def policy_iteration(g, gamma = 0.9, tol = 1e-6):
    # Uses policy iteration algorithm to get optimal policy

    # Set up random policy
    policy = g.actions.copy()
    for key, value in policy.items():
        probs = np.zeros(len(g.actions_array))
        value = np.random.choice(value)
        probs[g.actions_array == value] = 1
        policy[key] = probs

    # Conduct policy iteration
    policy_changed = True
    while (policy_changed):
        # print("Policy")
        # print_determinisitic_policy(policy, g)
        # print("\n")
        policy_changed = False
        # Get value function of current policy
        V = get_value(policy, g, gamma = gamma, tol = tol)
        # For each state, find the best action under the current value function
        for s in g.all_states(include_terminal=False):
            old_a = g.actions_array[policy[s] == 1][0]
            q = np.zeros(len(g.actions[s]))

            # get probability if random action is taken due to wind
            windy_prob = np.zeros(len(g.actions_array))
            windy_prob[np.isin(g.actions_array, g.actions[s])] = 1/len(g.actions[s])

            # Work out q for each action
            for i in range(len(g.actions[s])):
                # work out transition probabilities for this action in this state
                p = np.zeros(len(g.actions_array))
                p[g.actions_array == g.actions[s][i]] = 1
                p = p*(1-g.windy) + windy_prob*g.windy
                
                for j in range(len(p)):
                    # loop through possible end states given the desired action
                    if(p[j] != 0):
                        # take action if possible
                        g.set_state(s)
                        r = g.move(g.actions_array[j] ,force = True)
                        q[i] += p[j]*(r + gamma*V[g.current_state()])
            # get actions that maximimse q
            possible_a = g.actions[s][q == q.max()]
            new_a = old_a if old_a in possible_a else possible_a[0]
            # convert action into probability matrix
            policy[s] = np.zeros(len(g.actions_array))
            policy[s][g.actions_array == new_a] = 1
            # check if policy has changed
            policy_changed = policy_changed | (old_a != new_a)

    return((V, policy))

def print_determinisitic_policy(policy, g):
    out = np.full(shape = (g.height, g.width), fill_value = " ")
    for key, value in policy.items():
        action = g.actions_array[value == value.max()][0]
        out[key[0], key[1]] = action
    print(out)

def value_iteration(g, gamma = 0.9, tol = 1e-6, max_iter = 1000):
    # Uses value iteration algorithm to get optimal policy

    # Set up value function
    V = {}
    for s in g.all_states(include_terminal=True):
        V[s] = 0 # terminal states have 0 as value

    # Conduct value iteration
    not_small_enough = True
    iteration = 0
    while not_small_enough & (iteration < max_iter):
        delta = 0 

        # For each state, find the best action under the current value function
        for s in g.all_states(include_terminal=False):
            old_v = V[s]
            
            # Initialise Q
            q = np.zeros(len(g.actions[s]))

            # get probability if random action is taken due to wind
            windy_prob = np.zeros(len(g.actions_array))
            windy_prob[np.isin(g.actions_array, g.actions[s])] = 1/len(g.actions[s])

            # Work out q for each action
            for i in range(len(g.actions[s])):
                # work out transition probabilities for this action in this state
                p = np.zeros(len(g.actions_array))
                p[g.actions_array == g.actions[s][i]] = 1
                p = p*(1-g.windy) + windy_prob*g.windy
                
                for j in range(len(p)):
                    # loop through possible end states given the desired action
                    if(p[j] != 0):
                        # take action if possible
                        g.set_state(s)
                        r = g.move(g.actions_array[j] ,force = True)
                        q[i] += p[j]*(r + gamma*V[g.current_state()])
            # Update value function
            V[s] = q.max()

            # Check for convergence
            delta = max(delta, np.abs(V[s] - old_v))
        # End state loop - check for convergence
        not_small_enough = delta > tol
        iteration += 1

    # Get optimal policy
    policy = {}
    for s in g.all_states(include_terminal=False):
        q = np.zeros(len(g.actions[s]))

        # get probability if random action is taken due to wind
        windy_prob = np.zeros(len(g.actions_array))
        windy_prob[np.isin(g.actions_array, g.actions[s])] = 1/len(g.actions[s])

        # Work out q for each action
        for i in range(len(g.actions[s])):
            # work out transition probabilities for this action in this state
            p = np.zeros(len(g.actions_array))
            p[g.actions_array == g.actions[s][i]] = 1
            p = p*(1-g.windy) + windy_prob*g.windy
            
            for j in range(len(p)):
                # loop through possible end states given the desired action
                if(p[j] != 0):
                    # take action if possible
                    g.set_state(s)
                    r = g.move(g.actions_array[j] ,force = True)
                    q[i] += p[j]*(r + gamma*V[g.current_state()])
        # get actions that maximimse q
        action = g.actions[s][q == q.max()][0]
        # convert action into probability matrix
        policy[s] = np.zeros(len(g.actions_array))
        policy[s][g.actions_array == action] = 1


    return((V, policy))
