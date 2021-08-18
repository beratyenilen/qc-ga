from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
from qiskit.test.mock import FakeAthens

numberOfQubits = 5
NGEN = 100
POPSIZE = 100
stateIndex = 33
multiProcess = False
verbose = True
saveResult = True
# Let's try to use the basis gate of IBM Quantum Computers
#allowedGates = [X, SqrtX, CNOT, Rz, Swap]
allowedGates = [X, SqrtX, CNOT, Rz]
connectivity = FakeAthens().configuration().coupling_map

MAX_CIRCUIT_LENGTH = 10
epsilon = 0.0001
#allGates = [H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, Rz, Ry, Rx, SqrtX, Swap]
NEXT_STAGE_ERROR = 0.00002 # TOM: Changed from 0.12
