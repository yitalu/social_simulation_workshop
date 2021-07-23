import numpy as np

# global parameters
capacity = 20
n_players = 4
duration = 5
min_cow = 4
max_cow = 7
total_cow = 0
value_cow = 1
cost_cow = 1/n_players

# population array
players = np.zeros((n_players, 3)) # [0] types, [1] cows, [2] utility

# initialization
players[:, 0] = np.random.randint(0, 2, n_players) # 0: cooperator; 1: defector
players[:, 1] = np.random.randint(min_cow, max_cow, n_players)

for t in range(duration):
    total_cow = np.sum(players[:, 1])

    # agents make decisions
    for i in range(n_players):
        if players[i, 0] == 0: # cooperator
            players[i, 1] = players[i, 1] + 1 * (total_cow < capacity)
        elif players[i, 0] == 1: # defector
            players[i, 1] = players[i, 1] + 1
    
    total_cow = np.sum(players[:, 1])

    # calculate utility
    for i in range(n_players):
        players[i, 2] = value_cow * players[i, 1] - cost_cow * (total_cow - capacity) * (total_cow > capacity)
    
    print('time period:', t)
    print('id, types, cows, utility')
    for i in range(n_players):
        print(i, players[i, 0], players[i, 1], players[i, 2])

