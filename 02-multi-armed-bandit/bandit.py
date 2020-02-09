import numpy as np

class Bandit:
    def __init__(self, mu,  upper_limit):
        self.mu = mu
        self.mean = upper_limit
        self.N = 0

    def pull(self):
        reward = np.random.normal(loc = self.mu)
        self.N = self.N + 1
        self.mean = (1-1.0/self.N)*self.mean + 1.0/self.N*reward
        return reward

