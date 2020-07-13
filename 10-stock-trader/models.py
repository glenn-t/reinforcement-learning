#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:52:50 2020

@author: glenn

"""

import pdb
import numpy as np
import agents
from sklearn.preprocessing import StandardScaler

class FeatureGenerator:
    """Given a feature mapper function, returns a scaler"""
    def __init__(self, env, feature_mapper):
        
        
        # Same mapper
        self.feature_mapper = feature_mapper
        
        # Play one episode to generate data for scaler
        random_agent = agents.RandomAgent(env.n_stocks)
        X = []
        
        done = False
        state = env.reset()
        X.append(self.feature_mapper(state))
    
        while not done:
            action = random_agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state.copy()
            X.append(self.feature_mapper(state))
            
        # Create and fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        
    def generate_features(self, state):
        """
        Returns features from state

        Parameters
        ----------
        state : numpy array
            State vector
            

        Returns
        -------
        Transformed feature space

        """
        return(self.scaler.transform(self.feature_mapper(state)))
        

class LinearModel:
    
    def __init__(self, input_dim, n_action, alpha = 0.01):
        # Initiate weights
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        
        
    def predict(self):
        pass
        
    def sgd(self):
        pass
        
    def save(self):
        pass
    
    def load(self):
        pass
