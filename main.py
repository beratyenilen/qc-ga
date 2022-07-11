from numpy.ma.core import add
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
import numpy as np
from matplotlib import pyplot as plt
import os

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
import argparse

from qiskit import Aer, execute, QuantumRegister
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector, Operator
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeVigo, FakeAthens
from qiskit.circuit.library import Permutation
from qiskit_transpiler.transpiled_initialization_circuits import genCircs, getFidelities


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
problemName = f"{ID}-{NGEN}GEN-{stateName}"
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

def main():
    # Initialize your toolbox and population
    desired_state = loadState(numberOfQubits, stateIndex).data
    toolbox = initialize_toolbox(desired_state)
    pop = toolbox.population(n=POPSIZE)
    unaltered = LRSP_circs(desired_state, toolbox)
    for i in range(len(unaltered)):
        pop[i] = unaltered[i]

#    print(FILE_PATH)
#    if (load_file):
#        pop, log = load(FILE_PATH)
    
    start = time.perf_counter()
    pop, logbook = geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon=epsilon, verbose=verbose, returnLog=True)
    runtime = round(time.perf_counter() - start, 2)    
    
    x,y = plotCNOTSFidScatter(pop)
    plt.scatter(x,y)
    plt.show()
    paretoFront(pop)
    plt.show()

    # Save the results
    if saveResult:
        save(pop, logbook, directory, problemName)
        print(f"The population and logbook were saved in {directory}{problemName}")

    print(f'Runtime: {runtime}s')
    return runtime

if __name__ == '__main__':
    main()