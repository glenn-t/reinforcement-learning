# Given a policy, find the value function

import grid_world as gw
import numpy as np
import mc
import dynamic_programming_functions as dp

# Set up
g = gw.standard_grid()

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
actions = g.actions_array
for key, value in fixed_policy.items():
    probs = np.zeros(len(actions))
    probs[np.isin(actions, value)] = 1/len(value)
    fixed_policy[key] = probs

random_policy = g.actions.copy()
for key, value in random_policy.items():
    probs = np.zeros(len(actions))
    probs[np.isin(actions, value)] = 1/len(value)
    random_policy[key] = probs

# Policy evaluation using monte carlo
print("Policy evaluation examples:")
print("Windy grid world - MC")
g.windy = 0.5
mc.print_value_function(mc.get_value(g, fixed_policy, N=10000, gamma = 0.9), g)

print("Same using dynamic programming")
dp.print_value_function(dp.get_value(fixed_policy, g, gamma=0.9), g)
