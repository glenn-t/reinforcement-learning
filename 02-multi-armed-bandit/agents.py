import numpy as np

class epsilon_greedy:

    def __init__(self, n_bandits, eps):
        self.n_bandits = n_bandits
        self.means = np.repeat(0.0, n_bandits)
        self.bandit_N = np.repeat(0, n_bandits)
        self.eps = eps
        self.last_action = None
        self.first_run = True

    def choose_bandit(self):
        # This version explores the first turn and choose the max of only those it has explored.

        p = np.random.random()
        if self.first_run:
            j = np.random.choice(self.n_bandits)
            self.first_run = False
        elif p < self.eps:
            j = np.random.choice(self.n_bandits)
        else:
            ind_tried_bandits = self.bandit_N > 0
            # Take the max of bandits who we tried
            j = np.argmax(self.means[ind_tried_bandits])
            j = np.where(ind_tried_bandits)[0][j]
            j = np.argmax(self.means)
        
        self.last_action = j
        return j

    def update(self, reward):
        j = self.last_action
        self.bandit_N[j] = self.bandit_N[j] + 1
        self.means[j] = (1-1.0/self.bandit_N[j])*self.means[j] + 1.0/self.bandit_N[j]*reward

class adaptive_epsilon_greedy:
    # Uses decaying epsilon = 1/n

    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.means = np.repeat(0.0, n_bandits)
        self.bandit_N = np.repeat(0, n_bandits)
        self.last_action = None
        self.first_run = True

    def choose_bandit(self):
        # This version explores the first turn and choose the max of only those it has explored.

        p = np.random.random()
        if self.first_run:
            j = np.random.choice(self.n_bandits)
            self.first_run = False
        elif p < 1.0/(np.sum(self.bandit_N)+1):
            j = np.random.choice(self.n_bandits)
        else:
            ind_tried_bandits = self.bandit_N > 0
            # Take the max of bandits who we tried
            j = np.argmax(self.means[ind_tried_bandits])
            j = np.where(ind_tried_bandits)[0][j]
            j = np.argmax(self.means)
        
        self.last_action = j
        return j

    def update(self, reward):
        j = self.last_action
        self.bandit_N[j] = self.bandit_N[j] + 1
        self.means[j] = (1-1.0/self.bandit_N[j])*self.means[j] + 1.0/self.bandit_N[j]*reward

class optimistic_initial_values:

    def __init__(self, n_bandits, initial_value):
        self.n_bandits = n_bandits
        self.means = np.repeat(0.0, n_bandits)
        self.bandit_N = np.repeat(1, n_bandits)
        self.last_action = None
        self.means[:] = initial_value

    def choose_bandit(self):
        j = np.argmax(self.means)
        self.last_action = j
        return j

    def update(self, reward):
        j = self.last_action
        self.bandit_N[j] = self.bandit_N[j] + 1
        self.means[j] = (1-1.0/self.bandit_N[j])*self.means[j] + 1.0/self.bandit_N[j]*reward

class ucb1:

    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.means = np.repeat(0.0, n_bandits)
        self.bandit_N = np.repeat(0, n_bandits)
        self.last_action = None

    def choose_bandit(self):
        j = np.argmax(self.means + np.sqrt(2*np.log(np.sum(self.bandit_N) + 1)/(self.bandit_N+0.1)))
        self.last_action = j
        return j

    def update(self, reward):
        j = self.last_action
        self.bandit_N[j] = self.bandit_N[j] + 1
        self.means[j] = (1-1.0/self.bandit_N[j])*self.means[j] + 1.0/self.bandit_N[j]*reward

class bayesian:

    def __init__(self, n_bandits):
        self.n_bandits = n_bandits
        self.means = np.repeat(0.0, n_bandits)
        self.bandit_N = np.repeat(0, n_bandits)
        self.last_action = None

        # Bayesian parameters
        # Tau is precison of data - assumed
        # The smaller this is the lower the learning rate
        self.tau = 1

        # Set prior distribution - to be updated and will represent current posterior
        self.m0 = np.repeat(0.0, n_bandits)
        self.lambda0 = np.repeat(0.00001, n_bandits)

    def choose_bandit(self):
        posterior_samples = np.random.normal(loc = self.m0, scale = np.power(self.lambda0, -0.5))
        j = np.argmax(posterior_samples)
        self.last_action = j
        return j

    def update(self, reward):
        j = self.last_action
        self.bandit_N[j] = self.bandit_N[j] + 1
        self.m0[j] = (self.m0[j]*self.lambda0[j] + self.tau*reward)/(self.lambda0[j] + self.tau)
        self.lambda0[j] = self.lambda0[j] + self.tau
        # print("m0")
        # print(self.m0)
        # print("lambda0")
        # print(np.power(self.lambda0, -0.5))

