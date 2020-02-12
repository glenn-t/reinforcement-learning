# Glenn Thomas
# 2020-02-12
# Function to run different strategies for the bandit problem

import bandit
import agents
import numpy as np
import matplotlib.pyplot as plt

# Bandit means
mu = np.array([-1, 0, 1])
N = 100000
n_bandits = len(mu)

def run_experiment(mu, N, agent):
    """
    Runs the expirement

    Inputs:
    mu - numpy array of means for bandaits
    N - number of turns
    agent - a class of agent with methods choose_bandit and update.

    Output
    """
    # Make bandits
    n_bandits = len(mu)
    bandits = list()
    for i in range(n_bandits):
        bandits.append(bandit.Bandit(mu[i]))
   
    # Reward vector (could leave this to the agent)
    rewards = np.zeros(N)

    # Run simulation
    for i in range(N):
        j = agent.choose_bandit()
        reward = bandits[j].pull()
        agent.update(reward)
        rewards[i] = reward

    # Calculate average
    cumulative_average = np.cumsum(rewards)/(np.arange(N) + 1)
    return(cumulative_average)

### Epsilon greedy siumulation

for b in range(5):
    eps01 = run_experiment(mu, 10000, agents.epsilon_greedy(n_bandits, 0.01))
    eps10 = run_experiment(mu, 10000, agents.epsilon_greedy(n_bandits, 0.1))
    eps30 = run_experiment(mu, 10000, agents.epsilon_greedy(n_bandits, 0.3))
    if(b == 0):
        plt.plot(eps01, label='eps = 0.01', alpha = 0.3, color = "red")
        plt.plot(eps10, label='eps = 0.1', alpha = 0.3, color = "green")
        plt.plot(eps30, label='eps = 0.3', alpha = 0.3, color = "blue")
    else:
        plt.plot(eps01, alpha = 0.3, color = "red")
        plt.plot(eps10, alpha = 0.3, color = "green")
        plt.plot(eps30, alpha = 0.3, color = "blue")

plt.legend()
plt.xscale('log')
plt.savefig('output/comparing-epsilons.png')

eps10 = run_experiment(mu, N, agents.epsilon_greedy(n_bandits, 0.1))
eps_adaptive = run_experiment(mu, N, agents.adaptive_epsilon_greedy(n_bandits))
optimistic_initial_values = run_experiment(mu, N, agents.optimistic_initial_values(n_bandits, 10.0))
ucb1 = run_experiment(mu, N, agents.ucb1(n_bandits))
bayesian = run_experiment(mu, N, agents.bayesian(n_bandits))

plt.figure()
plt.plot(eps10, label = "Epsilon Greedy 0.1")
plt.plot(eps_adaptive, label = "Decaying Epsilon Greedy")
plt.plot(optimistic_initial_values, label = "Optimistic Initial Values")
plt.plot(ucb1, label = "UCB1")
plt.plot(bayesian, label = "Bayesian")
plt.legend()
plt.xscale('log')
plt.savefig('output/comparing_all_methods.png')
