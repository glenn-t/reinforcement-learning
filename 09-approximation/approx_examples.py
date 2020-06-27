# Glenn Thomas
# 2020-06-27
# Approximation examples

import grid_world as gw
import numpy as np
import approx_methods
import dynamic_programming_functions as dp
import matplotlib.pyplot as plt

# Set up
GAMMA = 0.9
g = gw.negative_grid(step_reward=0)
g.windy = 0.2

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
print("Windy grid world - MC with approx, step cost = 0")

def alpha(n):
    # If this is decaying, it can lead to extreme results
    # Seems more stable if using a contant, or use a low initial learning rate
    # return(0.05/np.sqrt(n))
    return(0.001)

dp.print_value_function(approx_methods.mc_predict(g, fixed_policy, alpha, N=10000, gamma = GAMMA), g)

print("Same using dynamic programming")
dp.print_value_function(dp.get_value(fixed_policy, g, gamma=GAMMA), g)