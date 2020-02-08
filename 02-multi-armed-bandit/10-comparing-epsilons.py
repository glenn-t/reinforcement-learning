import bandit
import numpy as np
import pdb

def run_experiment(mu, eps, N):
    # Make bandits
    n_bandits = len(mu)
    bandits = list()
    for i in range(n_bandits):
        bandits.append(bandit.Bandit(mu[i]))
    
    # Reward vector
    rewards = np.zeros(N)

    # Run simulation
    for i in range(N):
        p = np.random.random()
        if p < eps:
            j = np.random.choice(n_bandits) 
        else:
            means = np.array([b.mean for b in bandits])
            # If mean is zero it means we have no information
            # So randomly choose from the maximum one and the one that has zero. 
            # Has a caveat, if the maximum of the explored ones are negative, and there is zero mean, then that negative one will be excluded.
            # So more likley to explore a new one if you had chosen a negative one. 
            # options = np.where(np.logical_or(np.max(means) == means, means == 0))[0]
            # j = np.random.choice(options)

            # Course does it this way (below)
            j = np.argmax(means)
            
        rewards[i] = bandits[j].pull()
        
        #print(j)
        #print([b.mean for b in bandits])

    # Calculate average
    cumulative_average = np.cumsum(rewards)/(np.arange(N) + 1)
    return(cumulative_average)

# Run it
#result = run_experiment([-0.5, 0, 0.5], 0.1, 100)

# Plot result
B = 1000
result = np.zeros(B)
for b in range(B):
    result[b] = run_experiment([10, 10.5, 11], 0.15, 50)[-1:]

print(result.mean())
print(np.quantile(result, np.array([0.1, 0.9])))
