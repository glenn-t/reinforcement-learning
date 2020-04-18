# Given a policy, find the value function

from grid_world import standard_grid
import numpy as np
import pdb

SMALL_ENOUGH = 1e-6

# Set up
g = standard_grid()
states = g.all_states()

# Set up policies
fixed_policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'R',
    (2, 1): 'R',
    (2, 2): 'R',
    (2, 3): 'U',
  }
actions = np.array(["U", "D", "L", "R"])
for key, value in fixed_policy.items():
    probs = np.zeros(len(actions))
    probs[np.isin(actions, value)] = 1/len(value)
    fixed_policy[key] = probs

random_policy = g.actions.copy()
for key, value in random_policy.items():
    probs = np.zeros(len(actions))
    probs[np.isin(actions, value)] = 1/len(value)
    random_policy[key] = probs

# interative_policy_evaluation
def get_value(policy, gamma = 0.9):
    V = {}
    for s in g.all_states(include_terminal=True):
        V[s] = 0 # terminal states have 0 as value

    not_small_enough = True
    while not_small_enough:
        delta = 0
        for state in V.keys():
            old_V = V[state]
            # V(s) only has value if it's not a terminal state
            if state in g.actions:
                # bellman equations
                sum = 0
                # loop though all posible actions
                for i in range(len(policy[state])):
                    prob = policy[state][i]
                    if (prob > 0):
                        g.set_state(state)
                        #pdb.set_trace()
                        reward = g.move(actions[i])
                        sum += prob*(reward + gamma*V[g.current_state()])
                V[state] = sum
                delta = max(delta, np.abs(V[state] - old_V))
        not_small_enough = delta > SMALL_ENOUGH
    return(V)

def print_value_function(V):
    out = np.zeros((g.height, g.width))
    for key, value in V.items():
        out[key[0], key[1]] = value
    print(np.round(out, 2))
    return(out)

print("Random policy, gamma = 1")
print_value_function(get_value(random_policy, gamma = 1))

print("Fixed policy, gamma = 0.9")
print_value_function(get_value(fixed_policy, gamma = 0.9))


