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
        
        # Record metadata
        self.size = len(X[0])
        
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
        return(self.scaler.transform([self.feature_mapper(state)]))

class LinearModel:
    """
    LinearModel

    Parameters
    ----------
    input_dim : int
        Number of parameters.
    n_action : int
        Number of actions.
    alpha : float, optional
        Learning rate. The default is 0.01.
    momentum : float, optional
        Momentum rate - this speeds up convergence.
        Can cause the algortihm to be unstable unless paired
        with small enough alpha
        The default is 0.99.

    Returns
    -------
    None.

    """
        
    def __init__(self, input_dim, n_action, alpha = 0.01, momentum = 0.0):
        # Initiate weights
        self.W = np.random.random((input_dim, n_action)) / np.sqrt(input_dim)
        # Initiate bias weights
        self.b = np.random.random(n_action)
        self.alpha = alpha
        # Initialise momentum variables
        self.momentum = momentum
        self.W_velocity = np.zeros(self.W.shape)
        self.b_velocity = np.zeros(self.b.shape)
    
    def predict(self, X):
        return(np.matmul(X, self.W) + self.b)
        
    def sgd(self, y, X):
        
        normaliser = np.sum(X**2) + 1
        # Dividing by the above means alpha=1 and momentum=0 will
        # perfeclty adjust the model to fit the data
        # We add 1 because of the intercept term
        
        # Course said to do the below
        #num_vals = len(y)
        # Update W
        yhat = self.predict(X)
        # Not sure why we divide by num_vals (we do it in the course)
        grad_W = X.T.dot(yhat-y)/normaliser
        # Update momentum
        self.W_velocity = self.momentum*self.W_velocity - self.alpha*grad_W
        self.W = self.W + self.W_velocity
        
        # Update b
        grad_b = (yhat-y)/normaliser
        # Update momentum
        self.b_velocity = self.momentum*self.b_velocity - self.alpha*grad_b
        self.b = self.b + self.b_velocity
        
        mse = np.mean((yhat - y)**2)
        
    def save(self):
        pass
    
    def load(self):
        pass
