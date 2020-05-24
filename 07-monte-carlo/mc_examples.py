# Given a policy, find the value function

import grid_world as gw
import numpy as np
import mc
import dynamic_programming_functions as dp

# Set up
g = gw.negative_grid(step_reward=0)

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
print("Windy grid world - MC, step cost = 0")
g.windy = 0.5
mc.print_value_function(mc.get_value(g, random_policy, N=10, gamma = 1), g)

print("Same using dynamic programming")
dp.print_value_function(dp.get_value(random_policy, g, gamma=1), g)

print("Policy improvement examples")
# Policy improvement using monte carlo
V, policy = mc.mc_policy_improvement(g, gamma = 0.9, N = 3000)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

print("Same using dynamic programming")
V, policy = dp.policy_iteration(g, gamma = 0.9)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

