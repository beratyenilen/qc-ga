import random
import math
from copy import deepcopy
import numpy as np
from deap import tools
from deap.tools.emo import sortNondominated as sort_nondominated

from constants import *


def mutate_ind(individual):
    """Does one of 12 mutations to the circuit randomly
    """
    if individual.optimized:
        individual.parameter_mutation()
        return individual
    mutation_choice_fn = random.choice([
        individual.discrete_uniform_mutation,
        individual.continuous_uniform_mutation,
        individual.sequence_insertion,
        individual.sequence_and_inverse_insertion,
        individual.insert_mutate_invert,
        individual.sequence_deletion,
        individual.sequence_replacement,
        individual.sequence_swap,
        individual.sequence_scramble,
        individual.permutation_mutation,
        individual.trim,
        individual.move_gate
    ])
    mutation_choice_fn()
    return individual


def mate(parent1, parent2, toolbox):
    return (parent1.cross_over(parent2, toolbox), parent2.cross_over(parent1, toolbox))


def mutate_individuals(ranks, N, toolbox, current_rank=1):
    """This function takes a nested of list of individuals, sorted according to 
    their ranks and chooses a random individual. Each individual's probability
    to be chosen is proportional to exp(-(individual's rank)).
    """
    # FIXME what is the indexing of current_rank, clarify
    L = len(ranks)
    T = 0
    # Calculate the summation of exponential terms
    for i in range(L):
        T += math.exp(-current_rank-i)

    cps = []
    list_indexes = []
    element_indexes = []

    for _ in range(N):
        # Choose a random number between 0 and T
        random_number = random.uniform(0, T)
        # Find out which sublist this random number corresponds to
        # Let's say T = exp(-1) + exp(-2) + exp(-3) + exp(-4) and if our
        # random number is between 0 and 1, than it belongs to first list,
        # ranks[0]. If it is between exp(-1) and exp(-1)+exp(-2) than it
        # belongs to second list etc.

        # FIXME Refactor list index and everything else
        list_index = -1
        right_border = 0
        for i in range(L):
            right_border += math.exp(-current_rank-i)
            if random_number <= right_border:
                list_index = i
                break
        if list_index == -1:
            list_index = L-1
        left_border = right_border - math.exp(-current_rank-list_index)
        # Now, we will find out which index approximately the chosen number
        # corresponds to by using a simple relation.
        element_index = math.floor(
            len(ranks[list_index]) * (random_number-left_border)/(right_border-left_border))

        while len(ranks[list_index]) == 0:
            list_index += 1
            if len(ranks[list_index]) != 0:
                element_index = random.choice(range(len(ranks[list_index])))

        if element_index >= len(ranks[list_index]):
            element_index = -1
        # Copies the individual
        cp = deepcopy(ranks[list_index][element_index])
        cp = toolbox.mutate(cp)
        cp.fitness.values = toolbox.evaluate(cp)
        cps.append(cp)
        list_indexes.append(list_index)
        element_indexes.append(element_index)
    # FIXME remove Indexes from return
    return cps, list_indexes, element_indexes


def select_and_evolve(pop, toolbox):
    """
    We try to apply the selection and evolution procedure proposed in Potocek paper thing. 
    That is, we rank each circuit according via nondominated sorting then each 
    circuit is assigned a selection probability of exp(-r), where r is the rank.

    """
    # This function returns a list and ith element of the list contains the
    # individuals with rank i.
    to_carry = len(pop)//10
    individuals = toolbox.select(pop, to_carry)
    ranks = sort_nondominated(pop, len(pop))

#  paretoFront(pop)

    # Now we will carry the top 10% individuals to the next generation directly.
#  for indv in ranks[0]:
#      print(indv.fitness.values)

    next_generation = []
    for ind in individuals:
        next_generation.append(ind)

    # Now at this step I may loop over the chosen individuals and check if
    # two indvs have really close fitness values, less than let's say 0.1, and
    # remove one. However, I will skip this at this point.

    # We should assign a probability to choose each indv, w.r.t to their ranks.
    # Crossover prob. is 1/12 according to Potocek. We can increase it a little more.
    crossover = len(pop) // 12
    current_rank = 1
    N = len(pop)-to_carry - 2*crossover

    # TODO refactor chooseindividuals away: choose in this function and mutate with mutateindividual
    # individual_to_mutate = chooseindividualsToMutate(fronts, N, toolbox)
    individuals, _lis, _eis = mutate_individuals(
        ranks, N, toolbox, current_rank)
    next_generation.extend(individuals)

    for _ in range(crossover):  # refactor into function
        parent_index1 = random.randint(0, len(pop)-1)
        parent1 = pop[parent_index1]
        parent_index2 = random.randint(0, len(pop)-1)
        while parent_index1 == parent_index2:
            parent_index2 = random.randint(0, len(pop)-1)
        parent2 = pop[parent_index2]
        child1, child2 = toolbox.mate(parent1, parent2)
        child1.fitness.values = toolbox.evaluate(child1)
        child2.fitness.values = toolbox.evaluate(child2)
        next_generation.append(child1)
        next_generation.append(child2)

    return next_generation


def book_keep(best_candidate, output_file):
    output_file.write("\nbestCandidate has error:" +
                      str(best_candidate.fitness.values[0]))
    output_file.write("\nbestCandidate has circuit:")
    output_file.write(best_candidate.print_circuit())


def genetic_algorithm(pop, toolbox, number_of_generations, problem_name, problem_description):
    # Evaluate the individuals with an invalid fitness
    for ind in pop:
        if not ind.fitness.valid:
            ind.fitness.values = toolbox.evaluate(ind)

    output_file = open("./outputs/"+problem_name+".txt", "w")
    output_file.write(problem_description)
    print("Starting evolution, writing outputs to ./outputs/"+problem_name+".txt")
    # Register statistics functions to the toolbox
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_size = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)

    # Create the logbook
    logbook = tools.Logbook()
    # Start evolution
    for g in range(number_of_generations):
        print(g, "/", number_of_generations)
        # Retrieve the statistics for this generation.
        record = mstats.compile(pop)
        logbook.record(gen=g+1, **record)
        non_dominated_solutions = sort_nondominated(pop, len(pop))[0]

        # Select and evolve next generation of individuals
        next_generation = toolbox.select_and_evolve(pop, toolbox)
        # Evaluate the fitnesses of the new generation
        # We will evaluate the fitnesses of all the individuals just to be safe.
        # Mutations might be changing the fitness values, I am not sure.

        # The population is entirely replaced by the next generation of individuals.
        pop = next_generation
        best_candidate = non_dominated_solutions[0]
        for i in range(len(non_dominated_solutions)):
            if best_candidate.fitness.values[0] > non_dominated_solutions[i].fitness.values[0]:
                best_candidate = non_dominated_solutions[i]
        book_keep(best_candidate, output_file)

    logbook.header = "gen", "evals", "fitness", "size"
    logbook.chapters["fitness"].header = "min", "max", "avg", "std"
    logbook.chapters["size"].header = "min", "max", "avg", "std"
    print(logbook)

    return pop, logbook
