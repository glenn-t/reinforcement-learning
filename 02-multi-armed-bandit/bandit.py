import numpy as np

class Bandit:
    def __init__(self, mu,  upper_limit):
        # Enviroment varialbes (not known to strategy)
        self.__mu__ = mu

    def pull(self):
        reward = np.random.normal(loc = self.__mu__)
        return reward

