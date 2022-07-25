"""Configuration parameters for GA and noise simulation
"""
from projectq.ops import X, CNOT, Rz, SqrtX
import qiskit.providers.aer.noise as noise
from qiskit.test.mock import FakeVigo

# Number of qubits in the state
NUMBER_OF_QUBITS = 5

NUMBER_OF_GENERATIONS = 10
POPULATION_SIZE = 100
VERBOSE = True
SAVE_RESULT = True

# Backend configurations for noise simulation
FAKE_MACHINE = FakeVigo()
NOISE_MODEL = noise.NoiseModel.from_backend(FAKE_MACHINE)
BASIS_GATES = FAKE_MACHINE.configuration().basis_gates  # [X, SqrtX, CNOT, Rz]
ALLOWED_GATES = []

# TODO qiskit/text to projectq translate function
for g in BASIS_GATES:
    if g == 'x':
        ALLOWED_GATES.append(X)
    elif g == 'cx':
        ALLOWED_GATES.append(CNOT)
    elif g == 'sx':
        ALLOWED_GATES.append(SqrtX)
    elif g == 'rz':
        ALLOWED_GATES.append(Rz)
CONNECTIVITY = FAKE_MACHINE.configuration().coupling_map
MAX_CIRCUIT_LENGTH = 10
