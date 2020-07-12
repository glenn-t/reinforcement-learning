#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:56:27 2020

@author: glenn
"""

import pandas as pd
import numpy as np
import pdb

class StockMarket:
    
    def __init__(self, mode, initial_investment):
        assert mode in ["test", "train"]
        self.mode = mode
        self.initial_investment = initial_investment
        
        
        # Load data
        self.data = pd.read_csv("aapl_msi_sbux.csv")
        
        # Split into test/train. There is 1 day overlap (at the end of test)
        split_point = self.data.shape[0]//2
        if mode == "train":
            self.data = self.data.loc[0:split_point, ]
        else:
            self.data = self.data.loc[split_point:, ]
            
        # Convert data to matrix
        self.data = np.array(self.data)
        
        # Setup portfolio
        self.holdings = np.zeros(self.data.shape[1])
        self.state = np.zeros(self.data.shape[1]*2 + 1)
        
        # Setup initial variables
        self.reset()
        self.max_t = len(self.data) - 1
        
    def set_state(self):
        self.state[0:3] = self.prices
        self.state[3:6] = self.holdings
        self.state[6] = self.cash
        
    def reset(self):
        self.t = 0
        self.prices = self.data[self.t,:]
        self.holdings[:] = 0
        self.cash = self.initial_investment
        self.value = self.initial_investment
        self.done = False
        self.set_state()
        return(self.state)
        

    def step(self, action):
        assert not self.done
        assert type(action) is list
        assert len(action) == 3
        for a in action:
            assert a in ["BUY", "HOLD", "SELL"]
        
        ## First sell shares at current price
        for i, a in enumerate(action):
            if a == "SELL":
                self.cash = self.cash + self.holdings[i]*self.prices[i]
                self.holdings[i] = 0
                
        ## Second buy shares in round robin fashin
        if "BUY" in action:
           # Create indicator variable to check if bought this round
           bought = True
           while bought:
               bought = False
               for i, a in enumerate(action):
                   if (a == "BUY") and (self.prices[i] <= self.cash):
                       self.holdings[i] += 1
                       self.cash -= self.prices[i]
                       bought = True
                       
        ## Increment prices
        self.t = self.t + 1
        self.done = self.t == self.max_t
        self.prices = self.data[self.t,:]
        # Calculate new portfolio value
        new_value = np.sum(self.holdings*self.prices) + self.cash
        reward = new_value - self.value
        self.value = new_value
        self.set_state()
                       
        ## Return information
        return((self.state, reward, self.done, {"portfolio_value":self.value}))

       
       
    
                                       
                                       