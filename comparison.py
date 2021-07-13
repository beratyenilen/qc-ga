from math import perm
from qiskit_transpiler.transpiled_initialization_circuits import genCircs, getFidelities
from tools import load, loadState, plotCircLengths

from deap import creator, base, tools
from candidate import Candidate
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, transpile
from qiskit.tools.visualization import circuit_drawer, plot_circuit_layout, plot_histogram
from qiskit.test.mock import FakeVigo, FakeAthens
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector, Operator
from qiskit import BasicAer
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.extensions import snapshot_density_matrix
from qiskit.circuit.library import Permutation
from qiskit.transpiler import PassManager, CouplingMap, Layout
from qiskit.transpiler.passes import BasicSwap, LayoutTransformation, RemoveFinalMeasurements


def compare(pop, numberOfQubits, desired_state):
    initial_state = [0]*2**numberOfQubits
    initial_state[0] = 1
    circ = pop[0].toQiskitCircuit()
    circ.save_density_matrix()
    perm_unitary = pop[0].getPermutationMatrix()
    perm_desired_state = np.linalg.inv(perm_unitary) @ desired_state

    backend = Aer.get_backend('statevector_simulator')
    statevector = execute(circ, backend).result().get_statevector(circ)
#    print(state_fidelity(desired_state, perm_unitary @ statevector))
#    perm_desired_state = desired_state


    backend = Aer.get_backend('qasm_simulator')
    result = execute(circ,backend,shots=1).result()
    dens_matr = result.data()['density_matrix']

    print(len(dens_matr))
    print(len(dens_matr@perm_unitary))

    fid = state_fidelity(perm_desired_state, dens_matr)
    
    print(fid)
    print(circ)

    # Comparing the results with qiskit transpile funtion
    fake_machine = FakeAthens()
    qiskit_circs, depths = genCircs(numberOfQubits, fake_machine, desired_state)    
    circs = [circ.toQiskitCircuit() for circ in pop]
    circs = circs[0:1]
    plotCircLengths(qiskit_circs, circs)

def main():
    numberOfQubits = 3
    desired_state = loadState(numberOfQubits, 42)
    fitnessWeights = (-1.0,-1.0)
    creator.create("FitnessMin", base.Fitness, weights=fitnessWeights)
    creator.create("Individual", Candidate, fitness=creator.FitnessMin)
    path = 'saved/results/0-50GEN-3QB_state42.pop'
#    path = sys.argv[1]
    f = open(path, 'rb')
    pop = pickle.load(f)
    f.close()
    compare(pop, numberOfQubits, desired_state)

if __name__ == "__main__":
    main()

