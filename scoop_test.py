#   
#   An attempt to implement multiprocessing
#


# Importing the necessary modules.
import projectq
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
import numpy as np
import copy, sys, getopt, os

from deap import creator, base, tools
from candidate import Candidate
from constants import *
from new_evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm
from tools import *
from datetime import datetime
import time
import multiprocessing
import psutil

from qiskit import Aer, execute, QuantumRegister
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector, Operator
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeVigo, FakeAthens
from qiskit.circuit.library import Permutation
from qiskit_transpiler.transpiled_initialization_circuits import genCircs, getFidelities

def limit_cpu():
    "is called at every process start"
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(19)


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
        #return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
        return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
    else:
        return (error, 1.0)

import argparse


# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--POPSIZE", help = "Size of the population")
parser.add_argument("-g", "--NGEN", help = "The number of generations")
parser.add_argument("-q", "--NQUBIT", help = "The number of qubits")
parser.add_argument("-i", "--INDEX", help = "Index of desired state")

# Read arguments from command line
args = parser.parse_args()

if args.POPSIZE:
    POPSIZE = int(args.POPSIZE)
if args.NGEN:
    NGEN = int(args.NGEN)
if args.NQUBIT:
    numberOfQubits = int(args.NQUBIT)
if args.INDEX:
    stateIndex = int(args.INDEX)

stateName = str(numberOfQubits)+"QB_state"+str(stateIndex)
loadState(numberOfQubits, stateName)
now = datetime.now()
timeStr = now.strftime("%d.%m.%y-%H:%M")
#problemName = f"{timeStr}-{POPSIZE}pop-{NGEN}GEN-{stateName}"
directory = f"performance_data/{numberOfQubits}QB/{POPSIZE}POP/"
ID = int(len(os.listdir(directory)) / 2)
problemName = f"{ID}-{NGEN}GEN-{stateName}"

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

from scoop import futures

#if multiProcess:
#    pool = multiprocessing.Pool()
#    toolbox.register("map", pool.map)

toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", crossoverInd, toolbox=toolbox)
toolbox.register("mutate", mutateInd)
toolbox.register("select", tools.selNSGA2)
toolbox.register("selectAndEvolve", selectAndEvolve)
toolbox.register("evaluate", evaluateInd)

def main():
# Your main function
# epsilon is the error bound at which we simply finish the evolution and print out
# all the rank zero solutions.

# These probabilities were necessary if we were going to use the built-in
# selection and evolution algorithms, however since we have defined our own,
# we won't be using them.
    CXPB = 0.2
    MUTPB = 0.2

    
    #if len(sys.argv) > 1:
    #    path = sys.argv[1]
    #    f = open(path, 'rb')
    #    pop = pickle.load(f)
    #    f.close()
    #else:
    #    pop = toolbox.population(n=POPSIZE)

    pop = toolbox.population(n=POPSIZE)
    toolbox.register("map", futures.map)
#    toolbox.unregister("individual")
#    toolbox.unregister("population")

    start = time.perf_counter()
    pop, logbook = geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon, verbose=verbose, returnLog=True)
    finish = time.perf_counter()


#    plotFitSize(logbook)
#    plotFitSize(logbook, fitness="avg")
#    plotFitSize(logbook, fitness="std")

    print(evaluateInd(pop[0]))
    backend = Aer.get_backend('statevector_simulator')
    circ = pop[0].toQiskitCircuit()
    statevector = execute(circ, backend).result().get_statevector(circ)
    print(1 - state_fidelity(desiredState(), pop[0].getPermutationMatrix() @ statevector))


    # Matching the number of qubits with simulated machine
#    n_phys = 5
#    n = numberOfQubits
#    anc = QuantumRegister(n_phys-n, 'ancilla')
#    circ.add_register(anc)
#    aug_desired_state = desired_state
#    for k in range(n_phys-n):
#        aug_desired_state = np.kron([1,0],aug_desired_state)

#    from comparison import compare
#    compare(pop, numberOfQubits, desired_state)
     
    # Save the results
    if saveResult:
        save(pop, logbook, directory, problemName)
        print(f"The population and logbook were saved in {directory}{problemName}")

    print(f'Runtime: {round(finish-start, 2)}s')

if __name__=="__main__":
    main()
