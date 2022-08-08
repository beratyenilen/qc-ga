"""Configuration parameters for GA and noise simulation
"""

import qiskit.providers.aer.noise as noise
from qiskit.test.mock import FakeVigo
from tools import projectq_of_string

# number of qubits to be used for the problem
NUMBER_OF_QUBITS = 5

NUMBER_OF_GENERATIONS = 2000
POPULATION_SIZE = 200
SAVE_RESULT = True

# Backend configurations for noise simulation
FAKE_MACHINE = FakeVigo()
NOISE_MODEL = noise.NoiseModel.from_backend(FAKE_MACHINE)
# ['id', 'rz', 'sx', 'x', 'cx']
BASIS_GATES = FAKE_MACHINE.configuration().basis_gates

# allowed set of gates. Default is [Rz,SX,X,CX]
ALLOWED_GATES = [projectq_of_string(gate)
                 for gate in BASIS_GATES if gate != 'id']

CONNECTIVITY = FAKE_MACHINE.configuration().coupling_map
MAX_CIRCUIT_LENGTH = 10

#  A tuple describing the weight of each objective. A negative
# weight means that objective should be minimized, a positive weight means
# that objective should be maximized. For example, if you want to represent
# your weights as (error,circuitLen) and want to minimize both with equal
# weight you can just define fitnessWeights = (-1.0,-1.0). Only the relative
# values of the weights have meaning. BEWARE that they are multiplied and summed
# up while calculating the total fitness, so you might want to normalize them.
FITNESS_WEIGHTS = (-1.0, -0.5)
