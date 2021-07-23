import numpy as np
import sys
import os
import time
sys.path.append('/Users/yitalu/GoogleDrive/Notes/generic_models/')
import function_pop_dyn as func
import function_visualize as vsl

start = time.time()
os.chdir('figures')

# ------------------------------------------------
# START HERE: specify parameters, games, initial proportions, and dynamics here:
# ------------------------------------------------
# parameters
N = 400
T = 200 # generation
mr = 0.005 # mutation rate
sf = 1 # starting fitness
ar = 0.1 # adaptive learning rate (for BR)
pp = 0.5 # proportion of parents (for GA)
# try pp = 0.3, larger, and smaller

# game and initial proportions of types
# game, PR = 'hawk dove', [0.1, 0.9]
# ADD THIS: # game, PR = 'hawk dove bourgeois'
# game, PR = 'prisoner\'s dilemma', [1, 0]
game, PR = 'rps', [0.3, 0.3, 0.4]
# game, PR = 'pd with altruistic punisher', [0.7, 0.1, 0.2]

# choose dynamics
# dynamics = 'replicator'
# dynamics = 'best_response'
dynamics = 'genetic_algorithm'
# dynamics = 'projection'
# dynamics = 'monotone'

if game == 'hawk dove':
    types = ['Hawk', 'Dove']
    v, c = 4, 6 # benefit and cost in the payoff matrix
    M = np.array([ [(v-c)/2, v], [0, v/2] ]) # payoff matrix for the Hawk Dove game
if game == 'prisoner\'s dilemma':
    types = ['Cooperate', 'Defect']
    b, c = 5, 2 # benefit and cost in the payoff matrix (note: b > c)
    M = np.array([ [3, 0], [5, 1] ]) # payoff matrix
    M = np.array([ [b-c, -c], [b, 0] ]) # payoff matrix
if game == 'pd with altruistic punisher':
    types = ['Cooperate', 'Defect', 'Punish']
    M = np.array([ [2, 0, 2], [3, 1, -2], [2, -1, 2] ])
if game == 'rps':
    types = ['Rock', 'Paper', 'Scissors']
    M = np.array([ [1, 0, 2], [2, 1, 0], [0, 2, 1] ]) # payoff matrix for the RPS game


# ------------------------------------------------
# STOP HERE
# ------------------------------------------------


# population array and initialization
# [0] utility, [1] current type, [2] future type
pop = np.zeros((N, 3))
# types = np.zeros((ntypes, 4))
pop = func.create(pop, PR, sf) # func.create.__doc__
data = np.zeros((T+1, len(types)))
for i in range(len(types)):
    data[0, i] = np.sum(pop[:, 1] == i) / float(N)


# time loop
for t in range(T):
    pop = func.interact(pop, M, N)

    # replicator dynamics
    if dynamics == 'replicator':
        pop = func.replicator_dynamics(pop, types, sf)
    
    # best response dynamics with adaptive learning rate
    if dynamics == 'best_response':
        pop = func.best_response_dynamics(pop, types, M, ar)

    # genetic algorithm
    if dynamics == 'genetic_algorithm':
        pop = func.genetic_algorithm(pop, pp, sf)

    # mutation
    pop = func.mutate(pop, types, mr)
    
    # record data
    for i in range(len(types)):
        data[t+1, i] = np.sum(pop[:, 1] == i) / float(len(pop))

# plot
vsl.plot_line(data, T, types)
# vsl.plot_stacked_area(data, T, types)