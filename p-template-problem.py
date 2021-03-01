# Importing the necessary modules. 
import projectq
from projectq.ops import H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, Measure, All, Rx, Ry, Rz, SqrtX
import numpy as np
import copy
from constants import *
from deap import creator, base, tools
from candidate import Candidate
from constants import *
from evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm
# Import additional modules you want to use in here


def desiredState():
  '''
  This function returns the state vector of the desiredState as list where
  ith element is the ith coefficient of the state vector.
  '''
  return None


def evaluateInd(individual, verbose=False):
  '''
  This function should take an individual,possibly an instance of Candidate class, 
  and return a tuple where each element of the tuple is an objective.
  An example objective would be (error,circuitLen) where:
    error = |1 - < createdState | wantedState > 
    circuitLen = len(individual.circuit) / MAX_CIRCUIT_LENGTH
    MAX_CIRCUIT_LENGTH is the expected circuit length for the problem. 
  '''
  return (None, None)

# Your main function
if __name__ == "__main__":
  '''
  You should initialize:
    numberOfQubits : number of qubits to be used for the problem
    allowedGates   : allowed set of gates. Default is [Rz,SX,X,CX]
    problemName    : output of the problem will be stored at ./outputs/problemName.txt
    problemDescription : A small header describing the problem.
    fitnessWeights : A tuple describing the weight of each objective. A negative
      weight means that objective should be minimized, a positive weight means
      that objective should be maximized. For example, if you want to represent 
      your weights as (error,circuitLen) and want to minimize both with equal 
      weight you can just define fitnessWeights = (-1.0,-1.0). Only the relative 
      values of the weights have meaning. BEWARE that they are multiplied and summed 
      up while calculating the total fitness, so you might want to normalize them.
  '''
  # Initialize your variables
  numberOfQubits = 5
  allowedGates = [Rz,SqrtX,X,CNOT]
  problemName = "template"
  problemDescription = "Template Problem\nnumberOfQubits=" + str(numberOfQubits)+"\nallowedGates="+str(allowedGates)+"\n"
  fitnessWeights = (-1.0, -1.0)

  # Create the type of the individual
  creator.create("FitnessMin", base.Fitness, weights=fitnessWeights)
  creator.create("Individual", Candidate, fitness=creator.FitnessMin)
  # Initialize your toolbox and population
  toolbox = base.Toolbox()
  toolbox.register("individual", creator.Individual, numberOfQubits=numberOfQubits, allowedGates=allowedGates)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)
  # Register the necessary functions to the toolbox
  toolbox.register("mate", crossoverInd, toolbox=toolbox)
  toolbox.register("mutate", mutateInd)
  toolbox.register("select", tools.selNSGA2)
  toolbox.register("selectAndEvolve", selectAndEvolve)
  toolbox.register("evaluate", evaluateInd)

  # Get it running
  NGEN = 100      # For how many generations should the algorithm run ? 
  POPSIZE = 1000  # How many individuals should be in the population ? 
  verbose = False # Do you want functions to print out information. 
                  # Note that they will print out a lot of things. 
  
  # Initialize a random population
  pop = toolbox.population(n=POPSIZE)
  # Run the genetic algorithm
  geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon, verbose=verbose)
