from deap import creator, base, tools
from candidate import Candidate
from constants import *
from evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm

# import the a

def main():
  creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
  creator.create("Individual", Candidate, fitness=creator.FitnessMin)

  creator.create("name",)

  toolbox = base.Toolbox()
  toolbox.register("individual", creator.Individual)
  toolbox.register("population", tools.initRepeat, list, toolbox.individual)

  toolbox.register("mate", crossoverInd, toolbox=toolbox)
  toolbox.register("mutate", mutateInd)
  toolbox.register("select", tools.selNSGA2)
  toolbox.register("selectAndEvolve", selectAndEvolve)
  toolbox.register("evaluate", evaluateInd)

  toolbox.register("naem",)

  # LETS SEE IF IT WORKS
  NGEN = 500
  POPSIZE = 1000
  # epsilon is the error bound at which we simply finish the evolution and print out
  # all the rank zero solutions.
  
  # These probabilities were necessary if we were going to use the built-in
  # selection and evolution algorithms, however since we have defined our own,
  # we won't be using them.
  CXPB = 0.2
  MUTPB = 0.2

  pop = toolbox.population(n=POPSIZE)

  geneticAlgorithm(pop, toolbox, NGEN, epsilon, verbose=False)


