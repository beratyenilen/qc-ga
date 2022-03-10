from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
from qiskit.test.mock import FakeLima
import random

numberOfQubits = 5
NGEN = 1000
POPSIZE = 300
stateIndex = random.randint(1,100)
multiProcess = False
verbose = True
saveResult = True
# Let's try to use the basis gate of IBM Quantum Computers
#allowedGates = [X, SqrtX, CNOT, Rz, Swap]
fake_machine = FakeLima()
basis_gates = fake_machine.configuration().basis_gates #[X, SqrtX, CNOT, Rz]
allowedGates = []
for g in basis_gates:
    if g == 'x':
        allowedGates.append(X)
    elif g == 'cx':
        allowedGates.append(CNOT)
    elif g == 'sx':
        allowedGates.append(SqrtX)
    elif g == 'rz':
        allowedGates.append(Rz)
connectivity = fake_machine.configuration().coupling_map
#connectivity = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2]]
MAX_CIRCUIT_LENGTH = 10
epsilon = 0.01
#allGates = [H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, Rz, Ry, Rx, SqrtX, Swap]
NEXT_STAGE_ERROR = 0 # TOM: Changed from 0.12
