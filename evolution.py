import random
from candidate import Candidate
from deap.tools.emo import sortNondominated
from deap import tools
import math
import numpy as np
from constants import *
import pickle
from copy import deepcopy

def mutateInd(individual, verbose=False):
  if individual.fitness.values[0] < NEXT_STAGE_ERROR and not individual.optimized:
    individual.optimize()
    individual.parameterMutation()
    return individual, 
  if individual.optimized:
    individual.parameterMutatıon()
    return individual, 
  mutationChoice = random.choice(range(12))
  if mutationChoice == 0:
    individual.discreteUniformMutation()
  elif mutationChoice == 1:
    individual.continuousUniformMutation()
  elif mutationChoice == 2:
    individual.sequenceInsertion()
  elif mutationChoice == 3:
    individual.sequenceAndInverseInsertion()
  elif mutationChoice == 4:
    individual.insertMutateInvert()
  elif mutationChoice == 5:
    individual.swapQubits()
  elif mutationChoice == 6:
    individual.sequenceDeletion()
  elif mutationChoice == 7:
    individual.sequenceReplacement()
  elif mutationChoice == 8:
    individual.sequenceSwap()
  elif mutationChoice == 9:
    individual.sequenceScramble()
  elif mutationChoice == 10:
    individual.permutationMutation()
  else:
    individual.moveGate()
  return individual,

def crossoverInd(parent1, parent2, toolbox, verbose=False):
  child1 = toolbox.individual()
  child2 = toolbox.individual()
  Candidate.crossOver(parent1, parent2, child1)
  Candidate.crossOver(parent2, parent1, child2)
  return (child1, child2)

def chooseIndividual(ranks, currentRank=1, verbose=False):
  """
  This function takes a nested of list of individuals, sorted according to 
  their ranks and chooses a random individual. Each individual's probability
  to be chosen is proportional to exp(-(individual's rank)).
  """
  L = len(ranks)
  T = 0
  # Calculate the summation of exponential terms
  for i in range(L):
    T += math.exp(-currentRank-i)
  # Choose a random number between 0 and T
  randomNumber = random.uniform(0, T)
  # Find out which sublist this random number corresponds to
  # Let's say T = exp(-1) + exp(-2) + exp(-3) + exp(-4) and if our random number
  # is between 0 and 1, than it belongs to first list, ranks[0]. If it is
  # between exp(-1) and exp(-1)+exp(-2) than it belongs to second list etc.
  listIndex = -1
  rightBorder = 0
  for i in range(L):
    rightBorder += math.exp(-currentRank-i)
    if randomNumber <= rightBorder:
      listIndex = i
      break
  if listIndex == -1:
    listIndex = L-1
  leftBorder = rightBorder - math.exp(-currentRank-listIndex)
  # Now, we will find out which index approximately the chosen number corresponds
  # to by using a simple relation.
  elementIndex = math.floor(len(ranks[listIndex])*(randomNumber-leftBorder)/(rightBorder-leftBorder))

  while len(ranks[listIndex]) == 0:
    listIndex += 1
    if len(ranks[listIndex]) != 0:
      elementIndex = random.choice(range(len(ranks[listIndex])))

  if elementIndex >= len(ranks[listIndex]):
    elementIndex = -1
  cp = deepcopy(ranks[listIndex][elementIndex])
  return cp, listIndex, elementIndex

def selectAndEvolve(pop, toolbox, verbose=False):
  """
  We try to apply the selection and evolution procedure proposed in Potocek paper thing. 
  That is, we rank each circuit according via nondominated sorting then each 
  circuit is assigned a selection probability of exp(-r), where r is the rank.

  """
  # This function returns a list and ith element of the list contains the 
  # individuals with rank i.
  ranks = sortNondominated(pop, len(pop))
  # Now we will carry the top 10% individuals to the next generation directly.
  toCarry = int(len(pop)/10)
  nextGeneration = []
  bestCandidate = ranks[0][0]
  for rank in ranks:
    for indv in rank:
      if indv.fitness.values[0] < bestCandidate.fitness.values[0]:
        bestCandidate = indv
  nextGeneration.append(deepcopy(bestCandidate))
  currentRank = 0
  while len(nextGeneration) < toCarry:
    if (len(nextGeneration) + len(ranks[currentRank])) < toCarry:
      nextGeneration += deepcopy(ranks[currentRank])
      currentRank += 1
    else:
      chooseUpTo = toCarry - len(nextGeneration)
      nextGeneration += deepcopy(ranks[currentRank][:chooseUpTo])

  # Now at this step I may loop over the chosen individuals and check if
  # two indvs have really close fitness values, less than let's say 0.1, and 
  # remove one. However, I will skip this at this point. 

  # We should assign a probability to choose each indv, w.r.t to their ranks.
  probToMutate = 10/12  # Probability to perform mutation.
  # Crossover prob. is 1/12 according to Potocek. We can increase it a little more.
  currentRank = 1
  while len(nextGeneration) < len(pop):
    if random.random() <= probToMutate:
      # If this is the case, we'll mutate an individual and add it to nextGeneration
      individual,li,ei = chooseIndividual(ranks, currentRank, verbose)
      mutant, = toolbox.mutate(individual)
      nextGeneration.append(mutant)
    else:
      # If this is the case, we'll mate two individuals and add children to nextGeneration
      parent1,li1,ei1 = chooseIndividual(ranks, currentRank, verbose)
      parent2, li2,ei2 =  chooseIndividual(ranks, currentRank, verbose)
      while parent1 is parent2:
        parent2 = deepcopy(ranks[li2-1][ei2-1])
      child1, child2 = toolbox.mate(parent1, parent2)
      nextGeneration.append(child1)
      nextGeneration.append(child2)
  return nextGeneration

def terminateCondition(pop, toolbox, epsilon=0.001, verbose=False):
  """
  This function gets the rank 0 solutions and if there is a solution with error
  less than epsilon it returns True and all the non-dominated solutions.
  Returns:
    Bool, List of non-dominated solutions
  """
  nonDominatedSolutions = sortNondominated(pop, len(pop))[0]
  foundSolution = False
  for i in range(len(nonDominatedSolutions)):
    if nonDominatedSolutions[i].fitness.values[0] < epsilon:
      foundSolution = True
      break
  return foundSolution, nonDominatedSolutions

def bookKeep(bestCandidate, outputFile):
  outputFile.write("\nbestCandidate has error:" + str(bestCandidate.fitness.values[0]))
  outputFile.write("\nbestCandidate has circuit:")
  outputFile.write(bestCandidate.printCircuit(verbose=False))

def geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon=0.001, verbose=False, returnLog=False):
  # Evaluate the individuals with an invalid fitness
  invalid_ind = [ind for ind in pop if not ind.fitness.valid]
  fitnesses = map(toolbox.evaluate, invalid_ind) 
  for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
  
  outputFile = open("./outputs/"+problemName+".txt", "w")
  outputFile.write(problemDescription)
  print("Starting evolution, writing outputs to ./outputs/"+problemName+".txt")
  # Register statistics functions to the toolbox
  stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
  stats_size = tools.Statistics(key=lambda ind: ind.fitness.values[1]*MAX_CIRCUIT_LENGTH)
  mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)  
  mstats.register("avg", np.mean)
  mstats.register("std", np.std)
  mstats.register("min", np.min)
  mstats.register("max", np.max)
  
  # Create the logbook
  logbook = tools.Logbook()
  # Start evolution
  for g in range(NGEN):
    print(g,"/",NGEN)
    # Retrieve the statistics for this generation.
    record = mstats.compile(pop)
    logbook.record(gen=g+1, **record)
    _, nonDominatedSolutions = terminateCondition(pop, toolbox, epsilon, verbose)

    # Select and evolve next generation of individuals
    nextGeneration = toolbox.selectAndEvolve(pop, toolbox, verbose)
    # Evaluate the fitnesses of the new generation
    # We will evaluate the fitnesses of all the individuals just to be safe.
    # Mutations might be changing the fitness values, I am not sure.
    fitnesses = toolbox.map(toolbox.evaluate, nextGeneration)
    for ind, fit in zip(nextGeneration, fitnesses):
      ind.fitness.values = fit

    # The population is entirely replaced by the next generation of individuals.
    pop = nextGeneration
    bestCandidate = nonDominatedSolutions[0]
    for i in range(len(nonDominatedSolutions)):
      if bestCandidate.fitness.values[0] > nonDominatedSolutions[i].fitness.values[0]:
        bestCandidate = nonDominatedSolutions[i]
    bookKeep(bestCandidate, outputFile)
    
  logbook.header = "gen", "evals", "fitness", "size"
  logbook.chapters["fitness"].header = "min", "max", "avg", "std"
  logbook.chapters["size"].header = "min", "max", "avg", "std"
  print(logbook)
  gen = logbook.select("gen")
  fit_mins = logbook.chapters["fitness"].select("min")
  size_avgs = logbook.chapters["size"].select("avg")
  print(fit_mins)
  print(size_avgs)
  import matplotlib.pyplot as plt

  fig, ax1 = plt.subplots()
  line1 = ax1.plot(gen, fit_mins, "b-", label="Minimum Fitness")
  ax1.set_xlabel("Generation")
  ax1.set_ylabel("Fitness", color="b")
  for tl in ax1.get_yticklabels():
      tl.set_color("b")

  ax2 = ax1.twinx()
  line2 = ax2.plot(gen, size_avgs, "r-", label="Average Size")
  ax2.set_ylabel("Size", color="r")
  for tl in ax2.get_yticklabels():
      tl.set_color("r")

  lns = line1 + line2
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc="center right")

  plt.show()
  if returnLog:
      return pop, logbook
  return pop
