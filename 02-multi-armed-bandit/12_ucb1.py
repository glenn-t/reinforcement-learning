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

    # Run simulation
    for i in range(N):
        means = np.array([b.mean for b in bandits])
        # Add a small number to deal with the zero case
        attempts = np.array([b.N for b in bandits]) + 0.1
        j = np.argmax(means + np.sqrt(2*np.log(i + 1)/attempts))
        rewards[i] = bandits[j].pull()
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
        plt.plot(data, label='Optimistic Initial Values', alpha = 0.3, color = "red")
    else:
        plt.plot(data, alpha = 0.3, color = "red")
plt.legend()
plt.xscale('log')
plt.savefig('output/12-ucb1.png')

print(data[-1])
