#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:39:03 2020

@author: glenn
"""

import numpy as np
import models

# Actions are specified as "BUY", "HOLD" "SELL" for each stock
# Convert to a dictionary of all possible actions
action_mapping = {}
n_stocks = 3
possible_actions = ["BUY", "SELL", "HOLD"]
action_id = 0
for 

class RandomAgent:
    
    def __init__(self):
        pass
    
    def get_action(self, state):
         return(np.random.choice(["BUY", "SELL", "HOLD"], size = 3).tolist())
        
    def train(self, state, action, reward, next_state, done):
        pass

class LinearAgent:
    """Uses SARSA with a linear model"""
    
    def __init__(self, state_size, action_size):
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.01
        self.model = models.LinearModel(state_size, action_size, alpha)
    
    def get_action(self, state):
        if np.random.uniform(size=1) < self.epsilon:
            # Random action
            action = np.random.choice(["BUY", "SELL", "HOLD"], size = 3).tolist():
        else:    
            predictions = self.model.predict(state)
            
        
        
         return(np.random.choice(["BUY", "SELL", "HOLD"], size = 3).tolist())
        
    def train(self, state, action, reward, next_state, done):
        pass
    
    def load(self):
        pass
    
    def save(self):
        pass
