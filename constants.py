"""Configuration parameters for GA and noise simulation
"""
from projectq.ops import X, CNOT, Rz, SqrtX
import qiskit.providers.aer.noise as noise
from qiskit.test.mock import FakeVigo
from tools import projectq_of_string

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

# All basis gates are allowed
ALLOWED_GATES = [projectq_of_string(gate) for gate in BASIS_GATES]

CONNECTIVITY = FAKE_MACHINE.configuration().coupling_map
MAX_CIRCUIT_LENGTH = 10

# trying to minimize error and length
FITNESS_WEIGHTS = (-1.0, -0.5)
