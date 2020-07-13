#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:39:03 2020

@author: glenn
"""

import numpy as np
import models
import itertools

# Actions are specified as "BUY", "HOLD" "SELL" for each stock
# Convert to a dictionary of all possible actions
def get_all_actions(n_stocks):   
    action_mapping = {}
    possible_actions = ["BUY", "SELL", "HOLD"]
    for i, action in enumerate(itertools.product(possible_actions, repeat = n_stocks)):
        action_mapping[i] = list(action)
    return(action_mapping)


class RandomAgent:
    
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
    
    def get_action(self, state):
         return(np.random.choice(["BUY", "SELL", "HOLD"], size = self.n_stocks).tolist())
        
    def train(self, state, action, reward, next_state, done):
        pass

class LinearAgent:
    """Uses Q-learning with a linear approximation"""
    
    def __init__(self, n_stocks, feature_generator):
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01
        self.model = models.LinearModel(state_size, action_size, alpha)
        self.action_mapping = get_all_actions(n_stocks)
    
    def get_action(self, state):
        if np.random.uniform(size=1) < self.epsilon:
            # Random action
            action = np.random.choice(["BUY", "SELL", "HOLD"], size = 3).tolist()
        else:    
            predictions = self.model.predict(state)
            action_ind = np.argmax(predictions)[0]
            action = self.action_mapping[action_ind]
        
        return(action)

    def train(self, state, action, reward, next_state, done):
        pass
    
    def load(self):
        pass
    
    def save(self):
        pass
