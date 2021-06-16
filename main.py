# Importing the necessary modules.
import projectq
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
import numpy as np
import copy

from deap import creator, base, tools
from candidate import Candidate
from constants import *
from evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm
from tools import *
from datetime import datetime
import time
import multiprocessing
import sys


def loadState(numberOfQubits, stateName):
    f = open('states/'+str(numberOfQubits)+'_qubits/' + stateName, 'rb')
    global desired_state
    desired_state = pickle.load(f)
    f.close()

def desiredState():
    """
    This function returns the state vector of the desiredState as list where
    ith element is the ith coefficient of the state vector.
    """
    return desired_state


def evaluateIndcostt(individual, verbose=False):
    """
    This function should take an individual,possibly an instance of Candidate class,
    and return a tuple where each element of the tuple is an objective.
    An example objective would be (error,circuitLen) where:
      error = |1 - < createdState | wantedState >
      circuitLen = len(individual.circuit) / MAX_CIRCUIT_LENGTH
      MAX_CIRCUIT_LENGTH is the expected circuit length for the problem.
    """
    wanted = desiredState()
    got = individual.simulateCircuit()
    error = 1 - np.absolute(np.vdot(wanted, got))**2
    individual.setCMW(error)
    cost = individual.evaluateCost()
    if verbose:
        print("Wanted state is:", wanted)
        print("Produced state is", got)
        print("Error is:", error)
    return (error,cost)

def evaluateInd(individual, verbose=False):
    """
    This function should take an individual,possibly an instance of Candidate class,
    and return a tuple where each element of the tuple is an objective.
    An example objective would be (error,circuitLen) where:
      error = |1 - < createdState | wantedState >
      circuitLen = len(individual.circuit) / MAX_CIRCUIT_LENGTH
      MAX_CIRCUIT_LENGTH is the expected circuit length for the problem.
    """
    wanted = desiredState()
    got = individual.simulateCircuit()
    error = 1 - np.absolute(np.vdot(wanted, got))**2
    individual.setCMW(error)
    if verbose:
        print("Wanted state is:", wanted)
        print("Produced state is", got)
        print("Error is:", error)
    if len(individual.circuit) > 0 and len(individual.circuit) < MAX_CIRCUIT_LENGTH:
        return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
    else:
        return (error, 1.0)

from configparser import ConfigParser
def loadConfig(f):
    config = ConfigParser()
    config.read(f)
    print(config.sections())
    print(config['qubits'])
    
import json
def main():
    """
        This program expects config file as command line argument
    """

    if len(sys.argv)==1:
        print("No agument given!")
        return
    
    try:
        with open(sys.argv[1]) as f:
            data = json.load(f)
        numberOfQubits = data['qubits']
        NGEN = data['generations']
        POPSIZE = data['popsize']
        stateIndex = data['index']
        multiProcess = data['multiprocessing']
        verbose = data['verbose']
    except OSError:
        print("ERROR:   Could not open ", sys.argv[1])
        return

     
    # Initialize your variables
    #stateIndex = 42 
    stateName = str(numberOfQubits)+"QB_state"+str(stateIndex)
    loadState(numberOfQubits, stateName)
    now = datetime.now()
    timeStr = now.strftime("%d.%m.%y-%H:%M")
    ID = now.strftime("%d%m%y%H%M%S")+str(POPSIZE)+str(NGEN)+str(numberOfQubits)   #This needs improving
    problemName = f"{ID}-{timeStr}-{POPSIZE}pop-{NGEN}GEN-{stateName}"

# Let's try to use the basis gate of IBM Quantum Computers
    allowedGates = [X, SqrtX, CNOT, Rz, Swap]
    problemDescription = "State initalization for:\n"
    problemDescription += "numberOfQubits=" + str(numberOfQubits) + "\n"
    problemDescription += "allowedGates=" + str(allowedGates) + "\n"

# trying to minimize error and length !
    fitnessWeights = (-1.0, -1.0)

# Create the type of the individual
    creator.create("FitnessMin", base.Fitness, weights=fitnessWeights)
    creator.create("Individual", Candidate, fitness=creator.FitnessMin)
# Initialize your toolbox and population
    toolbox = base.Toolbox()

    if multiProcess:
        pool = multiprocessing.Pool()
        toolbox.register("map", pool.map)


    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossoverInd, toolbox=toolbox)
    toolbox.register("mutate", mutateInd)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("selectAndEvolve", selectAndEvolve)
    toolbox.register("evaluate", evaluateInd)

# Your main function
# epsilon is the error bound at which we simply finish the evolution and print out
# all the rank zero solutions.

# These probabilities were necessary if we were going to use the built-in
# selection and evolution algorithms, however since we have defined our own,
# we won't be using them.
    CXPB = 0.2
    MUTPB = 0.2

    
#    while (true):
#        response = input('load values? (y/n) ')
#        if response.lower() == 'y':
#            f = open('saved/results', 'rb')
#            pop = pickle.load(f)
#            f.close()
#            break
#        elif response.lower() == 'n':
#            break
    pop = toolbox.population(n=POPSIZE)

    start = time.perf_counter()
    pop, logbook = geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon, verbose=verbose, returnLog=True)
    finish = time.perf_counter()


    plotFitSize(logbook)
    plotFitSize(logbook, fitness="avg")
    plotFitSize(logbook, fitness="std")

    # Printing 10 best circuits
    #backend = Aer.get_backend('statevector_simulator')
    #for i in range(10):
    #    print(evaluateInd(pop[i]))
    #    circ = pop[i].toQiskitCircuit()
    #    statevector = execute(circ, backend).result().get_statevector(circ)
    #    print(1 - state_fidelity(desiredState(), pop[i].getPermutationMatrix() @ statevector))
        
    # Prompt to save the results
    #while (True):
    #    response = input('Save the values? (Y/N) ')
    #    if response.lower() == 'y':
    #        save(pop, logbook, "saved/test/", state_name=state_name)
    #        break
    #    elif response.lower() == 'n':
    #        break

    print(f'Runtime: {round(finish-start, 2)}s')

if __name__=="__main__":
    main()

