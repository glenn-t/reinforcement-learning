import numpy as np

class Bandit:
    def __init__(self, mu):
        self.mu = mu
        self.mean = 0
        self.N = 0

    def pull(self):
        reward = np.random.normal(loc = self.mu)
        self.N = self.N + 1
        self.mean = (1-1.0/self.N)*self.mean + 1.0/self.N*reward
        return reward

