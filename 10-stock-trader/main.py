#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 21:59:54 2020

@author: glenn
"""

import Stockmarket
import agents
import models
from helpers import play_game
import numpy as np
import matplotlib.pyplot as plt

INITIAL_INVESTMENT = 20000.0

train_env = Stockmarket.StockMarket("train", INITIAL_INVESTMENT)

def identity(x):
    return(x)

feature_generator = models.FeatureGenerator(train_env, identity)

random_agent = agents.RandomAgent(train_env.n_stocks)
linear_agent = agents.LinearAgent(train_env.n_stocks, feature_generator,gamma=0.95, epsilon_decay=0.9995, epsilon_min = 0.01, alpha = 0.01, momentum=0.9)

print(play_game(train_env, random_agent))


# Train
print("Training [LinearAgent, Random]")
N = 50
val = np.zeros((2, N))
for i in range(N):
    val[0, i] = play_game(train_env, linear_agent)
    val[1, i] = play_game(train_env, random_agent)
    print(val[:,i])

print("Average for Agent and Random")
print(np.mean(val, axis = 1))

# Test
print("Testing [LinearAgent, Random]")
N_test = 25
test_env = Stockmarket.StockMarket("test", INITIAL_INVESTMENT)
val_test = np.zeros((2, N_test))
for i in range(N_test):
    val_test[0, i] = play_game(test_env, linear_agent, train_flag = False)
    val_test[1, i] = play_game(test_env, random_agent, train_flag = False)
    print(val_test[:,i])

print("Average for Agent and Random")
print(np.mean(val_test, axis = 1))
