import numpy as np
import matplotlib.pyplot as plt

def create(population, proportions_types, starting_fitness):
    '''To generate a population with given proportions of each agent type.'''
    population[:, 0] = starting_fitness
    population[:, 1] = np.random.multinomial(1, proportions_types, len(population)).argmax(1)
    return population

def interact(population, payoff_matrix, number_agents):
    '''two randomly drawn agents interact with each other and get the payoffs specified in the payoff matrix'''
    random_matching = np.random.permutation(number_agents)
    for i in range( int(number_agents / 2) ):
        p1 = random_matching[i]
        p2 = random_matching[-i - 1]
        population[p1, 0] += payoff_matrix[int(population[p1, 1]), int(population[p2, 1])]
        population[p2, 0] += payoff_matrix[int(population[p2, 1]), int(population[p1, 1])]
    return population

def mutate(population, types, mutation_rate):
    number_agents = len(population)
    number_types = len(types)
    draw = np.random.uniform(0, 1, number_agents)
    mutants = np.where(draw < mutation_rate)
    number_mutants = len(mutants)
    population[mutants, 1] = np.random.randint(0, number_types, number_mutants)
    return population

def replicator_dynamics(population, types, starting_fitness):
    average_fitness = np.mean(population[:, 0])
    future_proportions = []

    for i in range(len(types)):
        number_type_i = np.sum(population[:, 1] == i)
        
        if number_type_i > 0:
            average_fitness_type_i = np.mean(population[population[:, 1] == i, 0])
            
            current_proportion_type_i = number_type_i / float(len(population))

            if average_fitness != 0:
                future_proportion_type_i = current_proportion_type_i * (average_fitness_type_i / average_fitness)
            elif average_fitness == 0:
                future_proportion_type_i = current_proportion_type_i
        
        elif number_type_i == 0:
            future_proportion_type_i = 0
        
        future_proportions.append(future_proportion_type_i)
        
    population = create(population, future_proportions, starting_fitness)
    return population

def best_response_dynamics(population, types, payoff_matrix, adaptive_learning_rate):
    '''Output the best (optimal/rational) response for adaptive learning. In adaptive learning, agents adapt its behavior in order to react to changes in proportions of agent types in the population.'''
    number_types = len(types)
    utility_types = np.zeros(number_types)
    
    for i in range(number_types):
        for j in range(number_types):
            number_type_j = np.sum(population[:, 1] == j)
            
            current_proportion_type_j = number_type_j / float(len(population))
            
            # the fitness of strategy i
            utility_types[i] += payoff_matrix[i, j] * current_proportion_type_j

    best_response = utility_types.argmax()
    draw = np.random.binomial(1, adaptive_learning_rate, len(population))
    population[np.where(draw == 1), 1] = best_response
    return population

def genetic_algorithm(population, proportion_of_parents, starting_fitness):
    current_generation = population[population[:, 0].argsort()][::-1] # sort by fitness (column 0)
    number_parents = int(len(current_generation) * proportion_of_parents)
    new_generation = np.zeros(current_generation.shape)
    
    for i in range(len(current_generation)):
        p1, p2 = np.random.randint(0, number_parents, 2) # parents 1 and 2
        new_generation[i, 0] = starting_fitness # starting fitness
        draw = np.random.randint(0, 2, 1) # draw either 0 or 1
        new_generation[i, 1] = current_generation[p1, 1] * draw + current_generation[p2, 1] * (1 - draw)  # weighted updating: 50% chance becomes parent 1's type and 50% chance becomes parent 2's type
    return new_generation