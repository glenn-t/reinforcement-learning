#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 21:23:19 2020

@author: glenn
"""


def play_game(env, agent, train_flag=True):

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
