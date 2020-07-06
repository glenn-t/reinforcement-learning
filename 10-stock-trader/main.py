#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:59:54 2020

@author: glenn
"""

import Stockmarket
import numpy as np

INITIAL_INVESTMENT = 20000.0

train_env = Stockmarket.StockMarket("test", INITIAL_INVESTMENT)
done = False
state = train_env.reset()
while not done:
    # Random agent
    action = np.random.choice(["BUY", "SELL", "HOLD"], size = 3).tolist()
    next_state, reward, done, info = train_env.step(action)
    state = next_state

print(info)
