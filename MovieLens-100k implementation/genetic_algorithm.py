import numpy as np
import random
import math
# from scipy.stats import stats


# Function to check if a profile exist in an array of profiles
def array_in_list(arr, list_arrays):
    return next((True for elem in list_arrays if np.array_equal(elem, arr)), False)


def wave(x, y):
    w = np.inner(x, y)

    # return 1 / (1 + math.exp(-w))
    return math.log((w + math.sqrt(1 + pow(w, 2))))


# function to calculate the average of a user's ratings
def calculate_average_rating(user):
    sum_of_ratings = 0
    number_of_ratings = 0
    for r in user:
        if r != 0:
            sum_of_ratings += r
            number_of_ratings += 1
    avg = sum_of_ratings / number_of_ratings

    return avg


def fitness(ind):
    fit = wave(ind, TARGET)
    return fit


def utility_matrix(population):
    matrix = np.zeros((len(population), len(population)))

    for i in range(len(population)):
        f1 = fitness(population[i])
        for j in range(0, i):
            f = wave(population[i], population[j])

            f2 = fitness(population[j])

            matrix[i, j] = (f1 + f2) * f

    order = np.dstack(np.unravel_index(np.argsort(matrix.ravel()), (len(population), len(population))))[0]
    order = np.flipud(order)

    sum_utility = matrix.sum()

    probabilities = []
    accumulated_prob = 0
    for i in order:
        x = i[0]
        y = i[1]

        accumulated_prob = accumulated_prob + matrix[x, y]/sum_utility
        probabilities.append(accumulated_prob)

        if probabilities[-1] > 1:
            break

    return order, probabilities


def choice_by_roulette(sorted_population, fitness_sum):
    normalized_fitness_sum = fitness_sum

    draw = random.uniform(0, 1)
    accumulated = 0
    for individual in sorted_population[:POPULATION]:
        fit = fitness(individual)
        probability = (fit / normalized_fitness_sum)
        accumulated += probability

        if draw <= accumulated:
            return individual

    return sorted_population[0]


def choice_by_game(population, pop_by_order, probabilities):

    r = random.uniform(0, 1)

    for i in range(len(probabilities)):

        if r < probabilities[i]:
            parentA = pop_by_order[i, 0]
            parentB = pop_by_order[i, 1]
            return population[parentA], population[parentB]

    if r > probabilities[-1]:
        parentA = pop_by_order[0, 0]
        parentB = pop_by_order[0, 1]
        return population[parentA], population[parentB]


def crossover(individual_a, individual_b):
    offspring1 = np.zeros(len(individual_a))
    offspring2 = np.zeros(len(individual_a))

    r1 = random.randint(0, len(individual_a) - 2)
    r2 = random.randint(r1, len(individual_a))

    offspring1[:r1] = individual_a[:r1]
    offspring2[:r1] = individual_b[:r1]

    offspring1[r1:r2] = individual_b[r1:r2]
    offspring2[r1:r2] = individual_a[r1:r2]

    offspring1[r2:] = individual_a[r2:]
    offspring2[r2:] = individual_b[r2:]

    return offspring1, offspring2


def mutate(individual):
    chrome = random.randint(0, len(individual) - 1)
    individual[chrome] = random.uniform(-1, 1)

    return individual


def make_next_generation(previous_population):
    # elitism
    next_generation = [previous_population[0]]

    if CHOICE == 1:
        pop_by_utility_order, probabilities = utility_matrix(previous_population)

    while len(next_generation) < POPULATION:

        if random.random() <= CROSS_PROP:

            if CHOICE == 0:

                fitness_sum = sum(fitness(individual) for individual in previous_population)
                first_choice = choice_by_roulette(previous_population, fitness_sum)
                second_choice = choice_by_roulette(previous_population, fitness_sum)

            else:

                parents = choice_by_game(previous_population, pop_by_utility_order, probabilities)
                first_choice = parents[0]
                second_choice = parents[1]

            individual1, individual2 = crossover(first_choice, second_choice)

            if random.random() <= MUTATE_PROP:
                individual1 = mutate(individual1)

            if random.random() <= MUTATE_PROP:
                individual2 = mutate(individual2)

            next_generation.append(individual1)

            next_generation.append(individual2)

        else:
            fitness_sum = sum(fitness(individual) for individual in previous_population)
            single_choice = choice_by_roulette(previous_population, fitness_sum)

            individual = single_choice

            if random.random() <= MUTATE_PROP:
                individual = mutate(individual)

            next_generation.append(individual)

    next_generation.sort(key=fitness, reverse=True)

    return next_generation


def get_first_gen(all_chromosomes):
    generation0 = []

    all_fitness = []
    for user in all_chromosomes:
        all_fitness.append(fitness(user))

    order = np.argsort(all_fitness)[::-1][:POPULATION]
    for i in order:
        generation0.append(all_chromosomes[i])

    return generation0


def run_ga(target, all_chromosomes):
    global TARGET
    TARGET = target

    generation = get_first_gen(all_chromosomes)
    old_generation = generation

    new_fitness_sum = sum(fitness(ind) for ind in generation)

    for i in range(GENERATIONS):
        print(new_fitness_sum)

        old_generation = generation
        generation = make_next_generation(generation)

        new_fitness_sum = sum(fitness(ind) for ind in generation)
        old_fitness_sum = sum(fitness(ind) for ind in old_generation)

        if new_fitness_sum <= old_fitness_sum:
            break

    return old_generation


# CHOICE = 0 for roulette and 1 for game
CHOICE = 1
POPULATION = 100
GENERATIONS = 100
CROSS_PROP = 0.7
MUTATE_PROP = 0.1
TARGET = []
