import projectq
from projectq.ops import H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, Measure, All, Rx, Ry, Rz
import numpy as np
import copy
from constants import *

from deap import creator, base, tools
from individual import Individual
from constants import *
from evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm


def createBellPair():
    eng = projectq.MainEngine()
    qureg = eng.allocate_qureg(2)
    H | qureg[0]
    CNOT | (qureg[0], qureg[1])
    eng.flush()
    mapping, wavefunction = copy.deepcopy(eng.backend.cheat())
    All(Measure) | qureg
    return wavefunction


def evaluateInd(individual, verbose=False):
    if (len(individual.circuit) >= MAX_CIRCUIT_LENGTH):
        del individual.fitness.values
        return (1.0, 1.0)
    finalState = individual.simulateCircuit()
    wantedState = createBellPair()
    dotProduct = np.vdot(wantedState, finalState)
    error = 1 - np.absolute(dotProduct)
    if verbose:
        print("\nfinalState is")
        print(finalState)
        print("wantedState is:")
        print(wantedState)
        print("dotProduct is:", dotProduct)
        print("error is then:", error)
    if len(individual.circuit) > 0:
        return (error, len(individual.circuit)/MAX_CIRCUIT_LENGTH)
    else:
        return (error, 1.0)


def main():

    numberOfQubits = 2
    #allowedGates = [Rx,Ry, Rz, CNOT]
    allowedGates = [X, H, S, CNOT, Y, Z]
    problemName = "bellpair"
    problemDescription = "Problem Name:" + problemName + "\n"
    problemDescription += "allowedGates: ["
    for i in range(len(allowedGates)):
        problemDescription += str(allowedGates[i]) + ", "
    problemDescription = problemDescription[:-2]
    problemDescription += "]\n"
    problemDescription += "numberOfQubits: " + str(numberOfQubits) + "\n"

    creator.create("fitness_min", base.Fitness, weights=(-1.0, -1.0))
    creator.create("individual", Candidate, fitness=creator.fitness_min)

    toolbox = base.Toolbox()
    toolbox.register("individual", creator.individual,
                     numberOfQubits=2, allowedGates=allowedGates)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossoverInd, toolbox=toolbox)
    toolbox.register("mutate", mutateInd)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("selectAndEvolve", selectAndEvolve)
    toolbox.register("evaluate", evaluateInd)

    # LETS SEE IF IT WORKS
    NGEN = 100
    POPSIZE = 1000

    verbose = False
    # epsilon is the error bound at which we simply finish the evolution and print out
    # all the rank zero solutions.

    # These probabilities were necessary if we were going to use the built-in
    # selection and evolution algorithms, however since we have defined our own,
    # we won't be using them.
    CXPB = 0.2
    MUTPB = 0.2

    pop = toolbox.population(n=POPSIZE)

    geneticAlgorithm(pop, toolbox, NGEN, problemName,
                     problemDescription, epsilon, verbose=verbose)


main()
