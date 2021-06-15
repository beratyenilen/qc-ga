# Importing the necessary modules.
import projectq
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
import numpy as np
import copy
from constants import *
from deap import creator, base, tools
from candidate import Candidate
from constants import *
from evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm

# Import additional modules you want to use in here
import pickle
from qiskit import Aer, execute
from qiskit.quantum_info import state_fidelity
from datetime import datetime
from tools import save

n = 3       #   Number of qubits
state_index = 42  
state_name = str(n)+'QB_state'+str(state_index)
f = open('states/'+str(n)+'_qubits/' + state_name, 'rb')
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


# Initialize your variables
numberOfQubits = n
# Let's try to use the basis gate of IBM Quantum Computers
allowedGates = [X, SqrtX, CNOT, Rz, Swap]
problemName = "kindaarbitrary"
problemDescription = "Kind of Arbitrary State initalization for:\n"
#problemDescription += str(c0) + "|00>" + str(c1) + "|01>" + str(c2) + "|10>" + str(c3) + "|11>\n"
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
toolbox.register("map", futures.map)

toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
# Register the necessary functions to the toolbox
toolbox.register("mate", crossoverInd, toolbox=toolbox)
toolbox.register("mutate", mutateInd)
toolbox.register("select", tools.selNSGA2)
toolbox.register("selectAndEvolve", selectAndEvolve)
toolbox.register("evaluate", evaluateInd)
# Your main function
if __name__ == "__main__":
    """
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
    """


    # Get it running
    NGEN = 100  # For how many generations should the algorithm run ?
    POPSIZE = 50 # How many individuals should be in the population ?
    verbose = False  # Do you want functions to print out information.
    # Note that they will print out a lot of things.

    # Initialize a random population
    pop = toolbox.population(n=POPSIZE)

    # Load values from previous run
    """
    while (True):
        response = input('Load values? (Y/N) ')
        if response.lower() == 'y':
            f = open('saved/results', 'rb')
            pop = pickle.load(f)
            f.close()
            break
        elif response.lower() == 'n':
            break
    """     

    # Run the genetic algorithm
    pop, logbook = geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon, verbose=verbose, returnLog=True)


    # Printing 10 best circuits
    backend = Aer.get_backend('statevector_simulator')
    for i in range(10):
        print(evaluateInd(pop[i]))
        circ = pop[i].toQiskitCircuit()
        statevector = execute(circ, backend).result().get_statevector(circ)
        print(1 - state_fidelity(desiredState(), pop[i].getPermutationMatrix() @ statevector))
        
    # Prompt to save the results
    while (True):
        response = input('Save the values? (Y/N) ')
        if response.lower() == 'y':
            save(pop, logbook, "saved/test/", state_name=state_name)
            break
        elif response.lower() == 'n':
            break
