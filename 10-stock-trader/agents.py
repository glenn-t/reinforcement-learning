#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:39:03 2020

@author: glenn
"""

import numpy as np
import models
import itertools
import pdb

# Actions are specified as "BUY", "HOLD" "SELL" for each stock
# Convert to a dictionary of all possible actions
def get_all_actions(n_stocks):   
    action_id_to_action = {}
    possible_actions = ["BUY", "SELL", "HOLD"]
    for i, action in enumerate(itertools.product(possible_actions, repeat = n_stocks)):
        action_id_to_action[i] = list(action)
        
    action_to_action_id = {}
    for key, value in action_id_to_action.items():
        action_to_action_id[tuple(value)] = key
    
    return((action_id_to_action, action_to_action_id))


class RandomAgent:
    
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
    
    def get_action(self, state):
         return(np.random.choice(["BUY", "SELL", "HOLD"], size = self.n_stocks).tolist())
        
    def train(self, state, action, reward, next_state, done):
        pass

class LinearAgent:
    """
    Uses Q-learning with a linear approximation
    
    Instead of approximating Q(s, a), we approximate Q(s) as a vector function returning a value for each a.
    """
    
    def __init__(self, n_stocks, feature_generator, gamma = 0.95, alpha = 0.1, epsilon = 1.0, epsilon_min = 0.001, epsilon_decay = 0.995, momentum=0.9):
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        self.feature_generator = feature_generator
        self.action_mapping, self.action_reverse_mapping = get_all_actions(n_stocks)
        self.n_actions = len(self.action_mapping)
        self.model = models.LinearModel(self.feature_generator.size, self.n_actions, alpha=alpha, momentum=momentum)
    
    def get_action(self, state):
        if np.random.uniform(size=1) < self.epsilon:
            # Random action
            action = np.random.choice(["BUY", "SELL", "HOLD"], size = 3).tolist()
        else:    
            predictions = self.model.predict(self.feature_generator.generate_features((state)))
            action_ind = np.argmax(predictions)
            action = self.action_mapping[action_ind]
        
        return(action)

    def train(self, state, action, reward, next_state, done):
        
        # Get action_id
        action_id = self.action_reverse_mapping[tuple(action)]
        # Generate target - for actions that weren't taken set target to 0
        X = self.feature_generator.generate_features((state))
        target = self.model.predict(X)
        
        # For action that was taken:
        if done:
            target[0, action_id] = reward
        else:
            V_next_state = np.max(self.model.predict(self.feature_generator.generate_features((next_state)))[0])
            target[0, action_id] = reward + self.gamma*V_next_state
            
        self.model.sgd(target, X)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

"""
TODO another idea for an agent is to have three models, one for each stock.
Each model either says buy, hold, sell.
This will reduce the number of variables to 8*3*n_stock = 72, down from 216 currently.
"""
