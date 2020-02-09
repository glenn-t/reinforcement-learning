import bandit
import numpy as np
import pdb
import matplotlib.pyplot as plt

# Implements optimisitic initial values

def run_experiment(mu, upper_limit, N):
    # Make bandits
    n_bandits = len(mu)
    bandits = list()
    for i in range(n_bandits):
        bandits.append(bandit.Bandit(mu[i], upper_limit))
    
    # Reward vector
    rewards = np.zeros(N)
    # Initialise bandit statistics
    means = np.repeat(0.0, n_bandits)
    # Apply the optimistic initial value
    means[:] = upper_limit
    # Need to set initial runs to 1
    bandit_N = np.repeat(1, n_bandits)

    # Run simulation
    for i in range(N):
        j = np.argmax(means)
        rewards[i] = bandits[j].pull()
        # Update mean
        bandit_N[j] = bandit_N[j] + 1
        means[j] = (1-1.0/bandit_N[j] )*means[j] + 1.0/bandit_N[j]*rewards[i]
    
       # print(j)
       # print([b.mean for b in bandits])

    # Calculate average
    cumulative_average = np.cumsum(rewards)/(np.arange(N) + 1)
    return(cumulative_average)

# Run it


# Plot it
for b in range(5):
    data = run_experiment(mu=[-1.0, 0.0, 1.0], upper_limit=10, N=100000)
    if(b == 0):
        plt.plot(data, label='Optimistic Initial Values', alpha = 0.3, color = "red")
    else:
        plt.plot(data, alpha = 0.3, color = "red")
plt.legend()
plt.xscale('log')
plt.savefig('output/11-optimistic_initial_values.png')

print(data[-1])
