import random
from candidate import Candidate
from deap.tools.emo import sortNondominated
import math

def mutateInd(individual, verbose=False):
  mutationChoice = random.choice(range(11))
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
  else:
    individual.moveGate()
  return individual,

def crossoverInd(parent1, parent2, toolbox, verbose=False):
  child1 = toolbox.individual()
  child2 = toolbox.individual()
  Candidate.crossOver(parent1, parent2, child1)
  Candidate.crossOver(parent1, parent2, child2)
  return (child1, child2)

def chooseIndividual(ranks, currentRank, verbose=False):
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
  if verbose:
    print("T:",T)
  # Choose a random number between 0 and T
  randomNumber = random.uniform(0, T)
  # Find out which sublist this random number corresponds to
  # Let's say T = exp(-1) + exp(-2) + exp(-3) + exp(-4) and if our random number
  # is between 0 and exp(-1), than it belongs to first list, ranks[0]. If it is
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
  if verbose:
    print("randomNumber:", randomNumber, ", leftBorder:", leftBorder, ", rightBorder:", rightBorder)
    print("listIndex: ", listIndex, " len(ranks[listIndex]): ", len(ranks[listIndex]), " elementIndex:", elementIndex)

  while len(ranks[listIndex]) == 0:
    listIndex += 1
    if len(ranks[listIndex]) != 0:
      elementIndex = random.choice(range(len(ranks[listIndex])))

  if elementIndex >= len(ranks[listIndex]):
    elementIndex = -1
  return ranks[listIndex][elementIndex]

def selectAndEvolve(pop, toolbox, verbose=False):
  """
  We try to apply the selection and evolution procedure proposed in Potocek paper thing. 
  That is, we rank each circuit according via nondominated sorting then each 
  circuit is assigned a selection probability of exp(-r), where r is the rank.

  """
  # This function returns a list and ith element of the list contains the 
  # individuals with rank i.
  ranks = sortNondominated(pop, len(pop))
  if verbose:
    print("Each rank includes this many individuals:")
    for i in range(len(ranks)):
      print(i, len(ranks[i]))

  # Now we will carry toCarry many best individuals to the next generation directly.
  toCarry = 100
  nextGeneration = []
  currentRank = 0
  flag = False
  while len(nextGeneration) < toCarry:
    if (len(nextGeneration) + len(ranks[currentRank])) < toCarry:
      nextGeneration += ranks[currentRank]
      currentRank += 1
    else:
      chooseUpTo = toCarry - len(nextGeneration)
      nextGeneration += ranks[currentRank][:chooseUpTo]
      ranks[currentRank] = ranks[currentRank][chooseUpTo:]
      flag = True
  if verbose:
    print("Length of nextGeneration: ", len(nextGeneration))
  
  if flag:
    ranks = ranks[currentRank:]
  else:
    ranks = ranks[(currentRank+1):]
  
  if verbose:
    for i in range(len(ranks)):
      print(len(ranks[i]))
  
  # Now at this step I may loop over the chosen individuals and check if
  # two indvs have really close fitness values, less than let's say 0.1, and 
  # remove one. However, I will skip this at this point. 

  # We should assign a probability to choose each indv, w.r.t to their ranks.
  probToMutate = 0.5  # Probability to perform mutation.
  while len(nextGeneration) < 1000:
    if random.random() <= probToMutate:
      # If this is the case, we'll mutate an individual and add it to nextGeneration
      individual = chooseIndividual(ranks, currentRank, verbose)
      mutant, = toolbox.mutate(individual)
      nextGeneration.append(mutant)
    else:
      # If this is the case, we'll mate two individuals and add children to nextGeneration
      parent1 = chooseIndividual(ranks, currentRank, verbose)
      parent2 =  chooseIndividual(ranks, currentRank, verbose)
      while parent1 is parent2:
        parent2 =  chooseIndividual(ranks, currentRank, verbose)
      child1, child2 = toolbox.mate(parent1, parent2)
      nextGeneration += [child1, child2]
  
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


def geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon=0.001, verbose=False):
  # Evaluate the individuals with an invalid fitness
  invalid_ind = [ind for ind in pop if not ind.fitness.valid]
  fitnesses = map(toolbox.evaluate, invalid_ind) 
  for ind, fit in zip(invalid_ind, fitnesses):
    ind.fitness.values = fit
  
  outputFile = open("./outputs/"+problemName+".txt", "w")
  outputFile.write(problemDescription)
  for g in range(NGEN):
    foundSolution, nonDominatedSolutions = terminateCondition(pop, toolbox, epsilon, verbose)
    '''
    # Allrighthy, Now I will comment out this section. I want GA to run NGEN
    # many generations and that is it. 
    if foundSolution:
      # This means we can terminate the genetic algorithm.
      outputFile.write("\nGenetic algorithm has found a solution in " + str(g+1) + " generations.\n")
      outputFile.write("We have "+str(len(nonDominatedSolutions))+" many non-dominated solutions.\n")
      for i in range(len(nonDominatedSolutions)):
        outputFile.write("\n"+str(i)+"th solution has error:"+str(nonDominatedSolutions[i].fitness.values[0])+'\n')
        outputFile.write(str(i)+"th solution has circuit:\n")
        outputFile.write(nonDominatedSolutions[i].printCircuit(verbose=False))
        outputFile.write("\n")
      
      return
    '''
    # Select and evolve next generation of individuals
    nextGeneration = toolbox.selectAndEvolve(pop, toolbox, verbose)
    # Evaluate the fitnesses of the new generation
    # We will evaluate the fitnesses of all the individuals just to be safe.
    # Mutations might be changing the fitness values, I am not sure.
    fitnesses = toolbox.map(toolbox.evaluate, nextGeneration)
    for ind, fit in zip(nextGeneration, fitnesses):
      ind.fitness.values = fit

    # The population is entirely replaced by the next generation of individuals.
    pop[:] = nextGeneration
    bestCandidate = nonDominatedSolutions[0]
    for i in range(len(nonDominatedSolutions)):
      if bestCandidate.fitness.values[0] > nonDominatedSolutions[i].fitness.values[0]:
        bestCandidate = nonDominatedSolutions[i]

    outputFile.write("\nbestCandidate has error:" + str(bestCandidate.fitness.values[0]))
    outputFile.write("\nbestCandidate has circuit:")
    outputFile.write(bestCandidate.printCircuit(verbose=False))

