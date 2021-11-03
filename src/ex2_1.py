#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def plot_belief(belief):
    
    plt.figure()
    
    ax = plt.subplot(2,1,1)
    ax.matshow(belief.reshape(1, belief.shape[0]))
    ax.set_xticks(np.arange(0, belief.shape[0],1))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_yticks([])
    ax.title.set_text("Grid")
    
    ax = plt.subplot(2, 1, 2)
    ax.bar(np.arange(0, belief.shape[0]), belief)
    ax.set_xticks(np.arange(0, belief.shape[0], 1))
    ax.set_ylim([0, 1.05])
    ax.title.set_text("Histogram")

# Action either string F or B. Belief is a belief vector of 15 positions. Returns belief.
def motion_model(action, belief):
    p_ff = p_bb = 0.7
    p_fb = p_bf = 0.1
    p_still = 0.2

    new_belief = np.zeros(len(belief))
    for i, b in enumerate(new_belief):
        # Update belief based on action taken. 
        if action == 'F':
            if i > 0 and i < 14:
                new_belief[i] = belief[i-1] * p_ff + belief[i] * p_still + belief[i+1] * p_fb
            elif i == 0:
                new_belief[i] = belief[i] * p_still + belief[i+1] * p_fb
            else:
                new_belief[i] = belief[i-1] * p_ff + belief[i] * p_still

        elif action == 'B':
            if i > 0 and i < 14:
                new_belief[i] = belief[i-1] * p_bf + belief[i] * p_still + belief[i+1] * p_bb
            elif i == 0:
                new_belief[i] = belief[i] * p_still + belief[i+1] * p_bb
            else:
                new_belief[i] = belief[i-1] * p_bf + belief[i] * p_still

    return new_belief

# Action either string F or B. Belief is a belief vector of 15 positions. Returns belief.
def motion_model2(action, belief):
    actions = {"CORR_DIR": 0.7, "NO_ACTION": 0.2, "INCORR_ACTION": 0.1}

    new_belief = np.zeros(len(belief))
    for i, b in enumerate(belief):

        # Update belief based on action taken. 
        if action == 'F':
            if(i < 14):
                new_belief[i+1] += b * actions["CORR_DIR"]
            if(i > 0):
                new_belief[i-1] += b * actions["INCORR_ACTION"]

            new_belief[i] += b * actions["NO_ACTION"]

        elif action == 'B':
            if(i > 0):
                new_belief[i-1] += b * actions["CORR_DIR"]
            if(i < 14):
                new_belief[i+1] += b * actions["INCORR_ACTION"] 
            new_belief[i] += b * actions["NO_ACTION"]

    return new_belief
    
def sensor_model(observation: int, belief, world):
    # Go through all locations and update belief based on observation. 
    for i, w in enumerate(world):
        # Black tile 
        if w == 0:
            p = 0.9 if w == observation else 0.1
            # Update belief
            belief[i] *= p
            continue

        # White tile 
        elif w == 1:
            p = 0.7 if w == observation else 0.3
            # Update belief
            belief[i] *= p
            continue

    return belief/sum(belief)


def recursive_bayes_filter(actions, observations, belief, world):
    # Initial position observation/sensor model
    belief = sensor_model(observations[0], belief, world)

    # Recursive prediction and correction step. (Motion model and sensor model/observation model)
    for action, observation in zip(actions, observations[1:]):
        belief = motion_model(action, belief)
        belief = sensor_model(observation, belief, world)
    
    return belief
