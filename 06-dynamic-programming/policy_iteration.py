# Given a policy, find the value function

from grid_world import standard_grid, negative_grid, big_grid, big_grid_negative
import dynamic_programming_functions as dp

# Set up
g = standard_grid()

print("Standard grid, gamma = 0.9")
V, policy = dp.policy_iteration(g, gamma = 0.9)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

print("Standard grid, gamma = 1")
V, policy = dp.policy_iteration(g, gamma = 1)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

g = negative_grid()
print("Negative grid, gamma = 0.9")
V, policy = dp.policy_iteration(g)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

print("Negative grid, gamma = 1")
V, policy = dp.policy_iteration(g, gamma = 1)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

g = big_grid()
print("Big grid, gamma = 1")
V, policy = dp.policy_iteration(g, gamma = 1)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)

g = big_grid_negative()
print("Big negative grid, gamma = 1")
V, policy = dp.policy_iteration(g, gamma = 1)
dp.print_value_function(V, g)
dp.print_determinisitic_policy(policy, g)