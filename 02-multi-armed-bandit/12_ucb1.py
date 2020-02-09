import bandit
import numpy as np
import pdb
import matplotlib.pyplot as plt

# Implements UCB1 - sets an upper bound based on number of trials

def run_experiment(mu, N):
    # Make bandits
    n_bandits = len(mu)
    bandits = list()
    for i in range(n_bandits):
        bandits.append(bandit.Bandit(mu[i],0))
    
    # Reward vector
    rewards = np.zeros(N)
    # Initialise bandit statis
    means = np.repeat(0.0, n_bandits)
    bandit_N = np.repeat(0, n_bandits)

    # Run simulation
    for i in range(N):
        j = np.argmax(means + np.sqrt(2*np.log(i + 1)/(bandit_N+0.1)))
        rewards[i] = bandits[j].pull()
        # Update mean
        bandit_N[j] = bandit_N[j] + 1
        means[j] = (1-1.0/bandit_N[j] )*means[j] + 1.0/bandit_N[j]*rewards[i]
        # TODO Using an initial value of zero - how does this bias the method?
    
       # print(j)
       # print([b.mean for b in bandits])

    # Calculate average
    cumulative_average = np.cumsum(rewards)/(np.arange(N) + 1)
    return(cumulative_average)

# Run it


# Plot it
for b in range(5):
    data = run_experiment(mu=[-1.0, 0.0, 1.0], N=100000)
    if(b == 0):
        plt.plot(data, label='UCB1', alpha = 0.3, color = "red")
    else:
        plt.plot(data, alpha = 0.3, color = "red")
plt.legend()
plt.xscale('log')
plt.savefig('output/12-ucb1.png')

print(data[-1])
