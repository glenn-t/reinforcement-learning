# Glenn Thomas
# 2020-06-27
# Approximation examples

import grid_world as gw
import numpy as np
import approx_methods
import dynamic_programming_functions as dp
import matplotlib.pyplot as plt
import td

# Reimport functions to aid in interactive development
import importlib
importlib.reload(approx_methods)

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
    return(0.1/np.sqrt(n))
    # return(0.01)
    # return(1/n)

def eps(n):
    # return(0.05/np.sqrt(n))
    return(1/np.sqrt(n))

dp.print_value_function(approx_methods.mc_predict(g, fixed_policy, alpha, N=1000, gamma = GAMMA), g)

print("Windy grid world - TD(0) with approx, step cost = 0")
# TD(0) does not fit so well in the off-policy regions 
# as it does not explore them as much as MC (using exploring starts)
dp.print_value_function(approx_methods.td0_predict(g, fixed_policy, alpha, N=1000, gamma = GAMMA, epsilon = 0.1), g)

print("Same using dynamic programming")
dp.print_value_function(dp.get_value(fixed_policy, g, gamma=GAMMA), g)

print("SARSA")

policy, value = approx_methods.sarsa(g=g, epsilon_function=eps, alpha_function = alpha, N = 1000, gamma = GAMMA)
td.print_value_function(value, g)
td.print_determinisitic_policy(policy, g)

print("Same using dynamic programming")
V, policy = dp.policy_iteration(g, gamma = 0.9)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)
