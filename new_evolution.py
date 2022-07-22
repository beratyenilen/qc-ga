import random
from candidate import Candidate
from deap.tools.emo import sortNondominated, selNSGA2
from deap.tools import selSPEA2
from deap import tools
import math
import numpy as np
from constants import *
import pickle
from copy import deepcopy


def mutateInd(individual, verbose=False):
  # if individual.fitness.values[0] < NEXT_STAGE_ERROR and not individual.optimized:
  #   individual.optimize()
  #   individual.parameterMutation()
  #   return individual
  if individual.optimized:
    individual.parameterMutation()
    return individual
  mutationChoice = random.choice(range(12))
  if mutationChoice == 0:
    individual.discreteUniformMutation()
  elif mutationChoice == 1:
    individual.continuousUniformMutation()
  elif mutationChoice == 2:
    individual.sequenceInsertion()
  # 3 and 4 cause the addition of sqrt(X)dagger gates
  elif mutationChoice == 3:
    individual.sequenceAndInverseInsertion()
  elif mutationChoice == 4:
    individual.insertMutateInvert()
  #elif mutationChoice == 5:
  #  individual.swapQubits()
  elif mutationChoice == 5:
    individual.sequenceDeletion()
  elif mutationChoice == 6:
    individual.sequenceReplacement()
  elif mutationChoice == 7:
    individual.sequenceSwap()
  elif mutationChoice == 8:
    individual.sequenceScramble()
  elif mutationChoice == 9:
    individual.permutationMutation()
  elif mutationChoice == 10:
    individual.trim()
  else:
    individual.moveGate()
  return individual

def crossoverInd(parent1, parent2, toolbox, verbose=False):
  child1 = toolbox.individual()
  child2 = toolbox.individual()
  Candidate.crossOver(parent1, parent2, child1)
  Candidate.crossOver(parent2, parent1, child2)
  return (child1, child2)

def chooseIndividuals(ranks, N, toolbox, currentRank=1, verbose=False):
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

  cps = []
  listIndexes = []
  elementIndexes = []
  
  for _ in range(N):
      # Choose a random number between 0 and T
      randomNumber = random.uniform(0, T)
      # Find out which sublist this random number corresponds to
      # Let's say T = exp(-1) + exp(-2) + exp(-3) + exp(-4) and if our random number
      # is between 0 and 1, than it belongs to first list, ranks[0]. If it is
      # between exp(-1) and exp(-1)+exp(-2) than it belongs to second list etc.

      # FIXME Refactor list index

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
      cp = deepcopy(ranks[listIndex][elementIndex]) # Copies the individual
      cp = toolbox.mutate(cp)
      cp.fitness.values = toolbox.evaluate(cp)
      cps.append(cp)
      listIndexes.append(listIndex)
      elementIndexes.append(elementIndex)
  # FIXME remove Indexes from return
  return cps, listIndexes, elementIndexes


from matplotlib import pyplot as plt
def selectAndEvolve(pop, toolbox, verbose=False):
  """
  We try to apply the selection and evolution procedure proposed in Potocek paper thing. 
  That is, we rank each circuit according via nondominated sorting then each 
  circuit is assigned a selection probability of exp(-r), where r is the rank.

  """
  # This function returns a list and ith element of the list contains the 
  # individuals with rank i.
  toCarry = len(pop)//10
  individuals = toolbox.select(pop, toCarry)
  ranks = sortNondominated(pop, len(pop))

#  paretoFront(pop)
  
  # Now we will carry the top 10% individuals to the next generation directly.
#  for indv in ranks[0]:
#      print(indv.fitness.values)

  nextGeneration = []
  for ind in individuals:
    nextGeneration.append(ind)

#  bestCandidate = ranks[0][0]
#  for rank in ranks:
#    for indv in rank:
#      if indv.fitness.values[0] < bestCandidate.fitness.values[0]:
#        bestCandidate = indv
#  nextGeneration.append(deepcopy(bestCandidate))
#
#  currentRank = 0
#  while len(nextGeneration) < toCarry:
#    if (len(nextGeneration) + len(ranks[currentRank])) < toCarry:
#      nextGeneration += deepcopy(ranks[currentRank])
#      currentRank += 1
#    else:
#      chooseUpTo = toCarry - len(nextGeneration)
#      nextGeneration += deepcopy(ranks[currentRank][:chooseUpTo])

  # Now at this step I may loop over the chosen individuals and check if
  # two indvs have really close fitness values, less than let's say 0.1, and 
  # remove one. However, I will skip this at this point. 

  # We should assign a probability to choose each indv, w.r.t to their ranks.
  # Crossover prob. is 1/12 according to Potocek. We can increase it a little more.
  crossover = len(pop) // 12
  currentRank = 1
  N=len(pop)-toCarry - 2*crossover

  individuals,lis,eis = chooseIndividuals(ranks, N, toolbox, currentRank, verbose)
  nextGeneration.extend(individuals)

  for _ in range(crossover):
      parentIndex1 = random.randint(0,len(pop)-1)
      parent1 = pop[parentIndex1]
      parentIndex2 = random.randint(0,len(pop)-1)
      while parentIndex1 == parentIndex2:
        parentIndex2 = random.randint(0,len(pop)-1)
      parent2 = pop[parentIndex2]
      child1, child2 = toolbox.mate(parent1, parent2)
      child1.fitness.values = toolbox.evaluate(child1)
      child2.fitness.values = toolbox.evaluate(child2)
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
  for ind in pop: 
    if not ind.fitness.valid:
      ind.fitness.values = toolbox.evaluate(ind) 
  
  outputFile = open("./outputs/"+problemName+".txt", "w")
  outputFile.write(problemDescription)
  print("Starting evolution, writing outputs to ./outputs/"+problemName+".txt")
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
  #print(fit_mins)
  #print(size_avgs)


  if returnLog:
      return pop, logbook
  return pop
