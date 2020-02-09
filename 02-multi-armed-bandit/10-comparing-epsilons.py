import bandit
import numpy as np
import pdb
import matplotlib.pyplot as plt

# Implements the epsilon greedy method

def run_experiment(mu, eps, N):
    # Make bandits
    n_bandits = len(mu)
    bandits = list()
    for i in range(n_bandits):
        bandits.append(bandit.Bandit(mu[i], upper_limit = 0))
    
    # Reward vector
    rewards = np.zeros(N)

    # Run simulation
    for i in range(N):
        p = np.random.random()
        # Explore on first run
        if np.logical_or(p < eps, i == 0):
            j = np.random.choice(n_bandits) 
        else:
            ind_tried_bandits = np.array([b.N for b in bandits]) != 0
            means = np.array([b.mean for b in bandits])
            
            # Take the max of bandits who we tried
            j = np.argmax(means[ind_tried_bandits])
            j = np.where(ind_tried_bandits)[0][j]
            
        rewards[i] = bandits[j].pull()
    
       # print(j)
       # print([b.mean for b in bandits])

    # Calculate average
    cumulative_average = np.cumsum(rewards)/(np.arange(N) + 1)
    return(cumulative_average)

# Run it


# Plot it
for b in range(5):
    eps1 = run_experiment([-1.0, 0.0, 1.0], 0.01, 10000)
    eps5 = run_experiment([-1.0, 0.0, 1.0], 0.1, 10000)
    eps10 = run_experiment([-1.0, 0.0, 1.0], 0.3, 10000)
    if(b == 0):
        print(eps1[-1])
        plt.plot(eps1, label='eps = 0.01', alpha = 0.3, color = "red")
        plt.plot(eps5, label='eps = 0.1', alpha = 0.3, color = "green")
        plt.plot(eps10, label='eps = 0.3', alpha = 0.3, color = "blue")
    else:
        plt.plot(eps1, alpha = 0.3, color = "red")
        plt.plot(eps5, alpha = 0.3, color = "green")
        plt.plot(eps10, alpha = 0.3, color = "blue")

plt.legend()
plt.xscale('log')
plt.savefig('output/10-comparing-epsilons01.png')

# Simulation of average performance

# B = 1000
# result = np.zeros(B)
# for b in range(B):
#     result[b] = run_experiment([-1, 0, 1], 0.01, 200)[-1:]
# print(result.mean())
# print(np.quantile(result, np.array([0.1, 0.9])))
