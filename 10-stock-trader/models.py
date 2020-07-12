#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 17:52:50 2020

@author: glenn

"""

import numpy as np

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
