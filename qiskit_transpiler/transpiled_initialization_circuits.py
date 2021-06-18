import matplotlib.pyplot as plt
import numpy as np
import math
from math import pi, factorial
from pprint import pprint

from numpy import absolute, vdot

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, Aer, transpile
#from qiskit.tools.visualization import circuit_drawer, plot_circuit_layout, plot_histogram
from qiskit.test.mock import FakeVigo, FakeAthens
from qiskit.quantum_info import state_fidelity, DensityMatrix, Statevector, Operator
from qiskit import BasicAer
#from qiskit.extensions import Initialize
#from qiskit.providers.aer import QasmSimulator
from qiskit.providers.aer.extensions import snapshot_density_matrix
from qiskit.circuit.library import Permutation
#from qiskit.transpiler import PassManager, CouplingMap, Layout
#from qiskit.transpiler.passes import BasicSwap, LayoutTransformation, RemoveFinalMeasurements

#   Computes the fidelities for transpiled circuits
def getFidelities(n, circs, machine_simulator, desired_vector):
    fidelities = []
    #n_shots = 20
    n_iter = len(circs)
    pad_vectors = []    #   List of perm_aug_desired_vectors for every circ
    for i in range(n_iter):
        
        qubit_pattern = list(circs[i]._layout.get_virtual_bits().values()) # How virtual bits map to physical bits
        n_phys = len(qubit_pattern) # n of physical bits

        #   Computing the permutations done by
        #   transpile function
        perm = [0,1,2,3,4]
        for op, qubits, clbits in circs[i].data:
            if op.name == 'measure':
                a = perm.index(clbits[0].index)
                b = perm[qubits[0].index]
                perm[qubits[0].index] = clbits[0].index
                perm[a] = b
        circs[i].remove_final_measurements()
        circs[i].save_density_matrix()
        
        qubit_pattern = perm

        aug_desired_vector = desired_vector

        for k in range(n_phys-n):
            aug_desired_vector = np.kron([1,0],aug_desired_vector)


        perm_circ = Permutation(n_phys, qubit_pattern) # Creating a circuit for qubit mapping
        perm_unitary = Operator(perm_circ) # Matrix for the previous circuit

        perm_aug_desired_vector = perm_unitary.data @ aug_desired_vector
        pad_vectors.append(perm_aug_desired_vector)

    for i in range(n_iter):
        result = execute(circs[i],machine_simulator,shots=1).result()
        noisy_dens_matr = result.data()['density_matrix']
        fid = state_fidelity(pad_vectors[i],noisy_dens_matr)
        fidelities.append(fid)
    return fidelities

#   Generate a random desired_vector
def randomDV(n):
    desired_vector = np.random.rand(2**n)+1j*np.random.rand(2**n)
    desired_vector = desired_vector/np.linalg.norm(desired_vector)
    return desired_vector


#   Generates n_iter amount of circuits for fake_machine
#   Returns lists for circuits and for depths
#
#   Measurement gates are added to keep track of the permutations
def genCircs(n, fake_machine, desired_vector, n_iter=10, optimization_level=2):
    qr = QuantumRegister(n)
    cr = ClassicalRegister(n)
    init_circ = QuantumCircuit(qr, cr)

    init_circ.initialize(desired_vector, qr)
    init_circ.barrier()
    init_circ.measure(qr, cr)

    circs = []
    depths = []

    for _ in range(n_iter):
        new_circ = transpile(init_circ,fake_machine,optimization_level=optimization_level)
        circs.append(new_circ)
        depths.append(new_circ.depth())
    return circs, depths

def main():
    machine_simulator = Aer.get_backend('qasm_simulator')
    fake_machine = FakeAthens()
    n = 4

    desired_vector = randomDV(n)
    print(desired_vector)
    print(np.linalg.norm(desired_vector))

    circs, depths = genCircs(n, fake_machine, desired_vector)

    fidelities = getFidelities(n, circs, machine_simulator, desired_vector)
    mean_fidelity = sum(fidelities)/len(fidelities)
    print("mean fidelity: " + str(mean_fidelity))

    plt.figure(figsize=(8, 6))
    plt.hist(fidelities, bins=list(np.arange(0,1.2,0.01)), align='left', color='#AC557C')
    plt.xlabel('Fidelity', fontsize=14)
    plt.ylabel('Counts', fontsize=14)
    plt.show()

if __name__ == "__main__":
    main()
