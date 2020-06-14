# Glenn Thomas
# 2020-06-01
# TD (Temporal differcing examples)

import grid_world as gw
import numpy as np
import td
import dynamic_programming_functions as dp

# Set up
g = gw.negative_grid(step_reward=-0.1)
g.windy = 0.5

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
print("Windy grid world - TD, step cost = -0.1")
dp.print_value_function(td.td0(g, fixed_policy, N=100, gamma = 0.9, alpha = 0.1, epsilon=0.1), g)

print("Same using dynamic programming")
dp.print_value_function(dp.get_value(fixed_policy, g, gamma=0.9), g)

print("Policy improvement examples")

def eps(N):
    # options 1/t, c/t, c/t^(a)
    # return(0.999**N) 
    return(1/N)

# There is the option to have decaying alpha, but would need to decay
# independantly for each element of Q, so need to track the count for each
# element of Q.

policy, value = td.sarsa(g=g, epsilon_function=eps, N = 100, gamma = 0.9, alpha=0.1)
td.print_value_function(value, g)
td.print_determinisitic_policy(policy, g)

print("Same using dynamic programming")
V, policy = dp.policy_iteration(g, gamma = 0.9)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

