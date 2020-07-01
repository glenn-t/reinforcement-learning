# Linear model for just the state (s)

import numpy as np

class Model_V:
  def __init__(self):
    self.theta = np.zeros(4)
  
  def generate_features(self, s):
    # Generate an X vector of features for a input state
    return np.array([1, s[0], s[1], s[0]*s[1]])

    # Generate using taylor polynomial
    # out = np.array([1.0, state[0], state[1], state[0]*state[1], state[0]**2, state[1]**2, state[0]*state[1]**2, state[0]**2*state[1], state[0]**3, state[1]**3, state[0]**2*state[1]**2])
    # Just use 4 variables (reducing from 9 states to learn, to 4 parameters to learn).
    # out = out[0:4]

    # If wanting to use all states - perfect model - one feature for each state
    # out = np.zeros(12)
    # cell_id = 0
    # for i in range(3):
    #     for j in range(4):
    #         out[cell_id] = state == (i,j)
    #         cell_id = cell_id + 1

  def predict(self, s):
    x = self.generate_features(s)
    return self.theta.dot(x)

  def grad(self, s):
    out = self.generate_features(s)
    # normalise direction
    size = np.sum(np.power(out,2))
    # Changing theta by out will increase the V_hat(s) by 1
    out = out/size
    return out

# Linear model for Q(a,a)

import numpy as np

class Model_Q:
  def __init__(self, g):
    self.actions_array = g.actions_array
    self.theta = np.zeros(4*len(self.actions_array))
  
  def generate_features(self, s, a):
    # Generate an X vector of features for a input state

    # 4 features with interaction on action
    features = np.zeros(4*len(self.actions_array))

    # Base features
    base_features = np.array([1, s[0], s[1], s[0]*s[1]])

    for i in range(len(self.actions_array)):
      hot_encoded_action = np.int((a == self.actions_array[i]))
      features[(4*i):(4*i + 4)] = base_features*hot_encoded_action

    return features

  def predict(self, s, a):
    x = self.generate_features(s, a)
    return self.theta.dot(x)

  def grad(self, s, a):
    out = self.generate_features(s, a)
    # normalise direction
    size = np.sum(np.power(out,2))
    # Changing theta by out will increase the V_hat(s) by 1
    out = out/size
    return out
