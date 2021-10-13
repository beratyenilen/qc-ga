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
from comparison import compare
import time
import multiprocessing
import psutil
import argparse

from qiskit import Aer, execute, QuantumRegister
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector, Operator
from qiskit.providers.aer import QasmSimulator
from qiskit.test.mock import FakeVigo, FakeAthens
from qiskit.circuit.library import Permutation
from qiskit_transpiler.transpiled_initialization_circuits import genCircs, getFidelities


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
        return (error, len(individual.circuit))
    else:
        return (error, len(individual.circuit))
        return (error, 1.0)


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

stateName = str(numberOfQubits)+"QB_state"+str(stateIndex)
loadState(numberOfQubits, stateName)
now = datetime.now()
timeStr = now.strftime("%d.%m.%y-%H:%M")
#problemName = f"{timeStr}-{POPSIZE}pop-{NGEN}GEN-{stateName}"
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

#from scoop import futures

#if multiProcess:
#    pool = multiprocessing.Pool()
#    toolbox.register("map", pool.map)


toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates, connectivity)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", crossoverInd, toolbox=toolbox)
toolbox.register("mutate", mutateInd)
toolbox.register("select", tools.selNSGA2)
toolbox.register("selectAndEvolve", selectAndEvolve)
toolbox.register("evaluate", evaluateIndcostt)

def main():
# Your main function
# epsilon is the error bound at which we simply finish the evolution and print out
# all the rank zero solutions.

# These probabilities were necessary if we were going to use the built-in
# selection and evolution algorithms, however since we have defined our own,
# we won't be using them.
    CXPB = 0.2
    MUTPB = 0.2

    tpop = toolbox.population(n=POPSIZE)
#    toolbox.register("map", futures.map)
#    toolbox.unregister("individual")
#    toolbox.unregister("population")

    pops = []
    n_circs = 100
    for _ in range(n_circs):
        start = time.perf_counter()
        pop, logbook = geneticAlgorithm(tpop, toolbox, NGEN, problemName, problemDescription, epsilon, verbose=verbose, returnLog=True)
        runtime = round(time.perf_counter() - start, 2)
        pops.append(pop)
    #-----------------------------------------------
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library import Permutation
    qubit_pattern = [0,1,2,3,4]
    perm_unitary = pop[0].getPermutationMatrix()
    perm_desired_state = np.linalg.inv(perm_unitary) @ desired_state
    n_phys = 5
    aug_desired_state = perm_desired_state
    for k in range(n_phys-numberOfQubits):
        aug_desired_state = np.kron([1,0],aug_desired_state)
    perm_circ = Permutation(n_phys, qubit_pattern) # Creating a circuit for qubit mapping
    perm_unitary = Operator(perm_circ) # Matrix for the previous circuit
    perm_aug_desired_state = perm_unitary.data @ aug_desired_state
    from qiskit.providers.aer.noise import NoiseModel
    from deap.tools.emo import sortNondominated
    machine_simulator = Aer.get_backend('qasm_simulator')
    fake_machine = FakeAthens()
    noise_model = NoiseModel.from_backend(fake_machine)
    coupling_map = fake_machine.configuration().coupling_map
    basis_gates = noise_model.basis_gates
    plt.figure(figsize=(8, 6))

    ranks = sortNondominated(pop, len(pop))
    front = ranks[0]
    fidelities = []
    for i in range(n_circs):
        circ = pops[i][0].toQiskitCircuit()
        s=0
        for _ in range(100):
            circ.snapshot_density_matrix('final')
            result = execute(circ,machine_simulator,
                            coupling_map=coupling_map,
                            basis_gates=basis_gates,
                            noise_model=noise_model,
                            shots=1).result()
            noisy_dens_matr = result.data()['snapshots']['density_matrix']['final'][0]['value']
            fid=state_fidelity(perm_aug_desired_state,noisy_dens_matr)
            s+=fid
        fidelities.append(s/100)
    plt.hist(fidelities, bins=list(np.arange(0,1.2,0.01)), align='left', label='GA')
    
    qiskit_circs, depths = genCircs(numberOfQubits, fake_machine, desired_state, n_iter=100)
    fidelities=getFidelities(5, qiskit_circs, machine_simulator, fake_machine, desired_state)
    #fidelities.append(s/100)
    plt.hist(fidelities, bins=list(np.arange(0,1.2,0.01)), align='left', color='#AC557C', label='Qiskit')
    plt.legend()
    print('!!!')
    plt.xlabel('Fidelity', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.savefig('100Fid_GA',dpi=300)
    
    plotFitSize(logbook)

    print(evaluateInd(pop[0]))
    backend = Aer.get_backend('statevector_simulator')
    circ = pop[0].toQiskitCircuit()
    statevector = execute(circ, backend).result().get_statevector(circ)
    print(state_fidelity(desiredState(), pop[0].getPermutationMatrix() @ statevector))
#    print(state_fidelity(pop[0].getPermutationMatrix() @ desiredState(), statevector))


#    compare(pop, numberOfQubits, desired_state)

     
    # Save the results
    if saveResult:
        save(pop, logbook, directory, problemName)
        print(f"The population and logbook were saved in {directory}{problemName}")

#    plotLenFidScatter(directory, problemName, numberOfQubits, stateName, evaluateInd, POPSIZE)
#    paretoFront(pop)

    print(f'Runtime: {runtime}s')
    return runtime

if __name__=="__main__":
    pop, logbook = load("performance_data/5QB/100POP/1-60000GEN-5QB_state33") 
    plotFitSize(logbook)
    paretoFront(pop)
#    main()
