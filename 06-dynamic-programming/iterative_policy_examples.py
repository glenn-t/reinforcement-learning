# Given a policy, find the value function

from grid_world import standard_grid
import numpy as np
import dynamic_programming_functions as dp

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

print("Random policy, gamma = 1")
dp.print_value_function(dp.get_value(random_policy, g, gamma = 1), g)

print("Fixed policy, gamma = 0.9")
dp.print_value_function(dp.get_value(fixed_policy, g, gamma = 0.9), g)


