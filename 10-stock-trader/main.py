#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:59:54 2020

@author: glenn
"""

import Stockmarket
import agents
import models
import numpy as np

INITIAL_INVESTMENT = 20000.0

def play_game(env, agent, train_flag = True):
    
    done = False
    state = env.reset()
    while not done:
        action = agent.get_action(state)
        next_state, reward, done, info = env.step(action)
        if train_flag:
            agent.train(state, action, reward, next_state, done)
        state = next_state

    # Return final portfolio value
    return(info['portfolio_value'])

train_env = Stockmarket.StockMarket("test", INITIAL_INVESTMENT)
random_agent = agents.RandomAgent(train_env.n_stocks)

print(play_game(train_env, random_agent))

def identity(x):
    return(x)

feature_generator = models.FeatureGenerator(train_env, identity)
print(feature_generator.generate_features(np.array([30, 30, 30, 30,30,30, 5000]).reshape(1, -1)))
