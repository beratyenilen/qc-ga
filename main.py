from numpy.ma.core import add
import projectq
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
import numpy as np
from matplotlib import pyplot as plt
import copy, sys, getopt, os

from deap import creator, base, tools
from qiskit.providers.aer.noise.errors.errorutils import circuit2superop
from qiskit_transpiler.transpiled_initialization_circuits import genCircs, getPermutation, getFidelities, randomDV
from candidate import qasm2ls
from candidate import Candidate
from constants import *
from new_evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm
from tools import *
from datetime import datetime
from comparison import compare
import time
import psutil
import argparse

from qiskit import Aer, execute, QuantumRegister
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector, Operator
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeVigo, FakeAthens
from qiskit.circuit.library import Permutation
from qiskit_transpiler.transpiled_initialization_circuits import genCircs, getFidelities

from qclib.state_preparation.baa_schmidt import initialize

desired_state = loadState(numberOfQubits, stateIndex)

def desiredState():
    """
    This function returns the state vector of the desiredState as list where
    ith element is the ith coefficient of the state vector.
    """
    return desired_state.data


def avgCNOTs(pop):
    avg_cnots = 0
    total = 0
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    for ind in front:
        circ = ind.circuit
        l = len(circ)
        if (l==0) :
            continue
        cnots = 0
        for gate in circ:
            if (gate[1]==CNOT): cnots+=1
        print(cnots/l)
        avg_cnots += cnots/l
        total += 1
    avg_cnots = avg_cnots/total
    return avg_cnots


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
    return (error, len(individual.circuit))
#    if len(individual.circuit) > 0 and len(individual.circuit) < MAX_CIRCUIT_LENGTH:
#        return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
#    else:
#        return (error, 1.0)


directory = f"performance_data/{numberOfQubits}QB/{POPSIZE}POP/"
ID = int(len(os.listdir(directory)) / 2)

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--POPSIZE", help = "Size of the population")
parser.add_argument("-g", "--NGEN", help = "The number of generations")
parser.add_argument("-q", "--NQUBIT", help = "The number of qubits")
parser.add_argument("-i", "--INDEX", help = "Index of desired state")
parser.add_argument("-id", "--ID", help = "ID of the saved file")

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
if args.ID:
    ID = int(args.ID)
    load_file = True

stateName = str(numberOfQubits)+"QB_state"+str(stateIndex)
#problemName = f"{ID}-{NGEN}GEN-{stateName}"
#FILE_PATH = f"performance_data/{numberOfQubits}QB/{POPSIZE}POP/"+problemName

now = datetime.now()
timeStr = now.strftime("%d.%m.%y-%H:%M")

problemDescription = "State initalization for:\n"
problemDescription += "numberOfQubits=" + str(numberOfQubits) + "\n"
problemDescription += "allowedGates=" + str(allowedGates) + "\n"

# trying to minimize error and length !
fitnessWeights = (-1.0, -0.5)

# Create the type of the individual
creator.create("FitnessMin", base.Fitness, weights=fitnessWeights)
creator.create("Individual", Candidate, fitness=creator.FitnessMin)

# Initialize your toolbox and population
toolbox = base.Toolbox()

toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates, connectivity)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", crossoverInd, toolbox=toolbox)
toolbox.register("mutate", mutateInd)
toolbox.register("select", tools.selSPEA2)
toolbox.register("selectAndEvolve", selectAndEvolve)
#toolbox.register("evaluate", evaluateIndcostt)
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

    pop = toolbox.population(n=POPSIZE)
        
#---------------------------------
    # define list of fidelity loss values to try out
    losses = list(np.linspace(0.0,1.0,100))
    # find the exact circuit
    circuit = initialize(desiredState(), max_fidelity_loss=0.0, strategy="brute_force", use_low_rank=True)
    circuit.measure_all()
    transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=3)

    # create a list of circuits with increasing fidelity loss
    circuits = [transpiled_circuit]

    for loss in losses:
        # find approximate initialization circuit with fidelity loss
        circuit = initialize(desiredState(), max_fidelity_loss=loss, strategy="brute_force", use_low_rank=True)
        circuit.measure_all()
        transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=3)
    
        #if transpiled_circuit.depth() < circuits[-1].depth():
        circuits.append(transpiled_circuit)
#---------------------------------

    #qiskit_circs, depths = genCircs(numberOfQubits, fake_machine, desiredState(), n_iter=1, measuregates=True)
    unaltered = []
    for i in range(len(circuits)):
        perm = getPermutation(circuits[i])
        qiskit_circ = qasm2ls(circuits[i].qasm())
        pop[i].circuit = qiskit_circ
        pop[i].permutation = perm
        pop[i].fitness.values = toolbox.evaluate(pop[i])
        unaltered.append(copy.deepcopy(pop[i]))
    
    print(FILE_PATH)
    if (load_file):
        pop, log = load(FILE_PATH)
    
    start = time.perf_counter()
    pop, logbook = geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon=epsilon, verbose=verbose, returnLog=True)
    runtime = round(time.perf_counter() - start, 2)

    avg_cnots = avgCNOTs(pop)

    paretoFront(pop, numberOfQubits, color='red', all=False)
    paretoFront(unaltered, numberOfQubits, color='blue', all=False)
    plt.show()
    plotLenCNOTScatter(pop, color='red')
    plotLenCNOTScatter(unaltered, color='blue')
    plt.show()
    paretoNoiseFids(pop, numberOfQubits, desiredState(), noise_model, avg_cnot=avg_cnots, connectivity_len=len(connectivity), all=False)
    plt.show()
    print(evaluateInd(pop[0]))
    backend = Aer.get_backend('statevector_simulator')
    circ = pop[0].toQiskitCircuit()
    statevector = execute(circ, backend).result().get_statevector(circ)
#    print(state_fidelity(desiredState(), pop[0].getPermutationMatrix() @ statevector))
#    print(state_fidelity(pop[0].getPermutationMatrix() @ desiredState(), statevector))

    
    
    # Save the results
    if saveResult:
        save(pop, logbook, "", "TEST")
        print(f"The population and logbook were saved in {directory}{problemName}")

    print(f'Runtime: {runtime}s')
    return runtime


if __name__=="__main__":
    main()