#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:56:27 2020

@author: glenn
"""

import pandas as pd

class StockMarket:
    
    def __init__(self, mode):
        self.mode = mode
        assert mode in ["test", "train"]
        
        # Load data
        self.data = pd.read_csv("aapl_msi_sbux.csv")
        
        # Split into test/train. There is 1 day overlap (at the end of test)
        split_point = self.data.shape[0]//2
        if mode == "train":
            self.data = self.data.loc[0:split_point, ]
        else:
            self.data = self.data.loc[(split_point:, ]
