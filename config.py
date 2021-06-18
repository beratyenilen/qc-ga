from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap

numberOfQubits = 5
NGEN = 100
POPSIZE = 50
stateIndex = 42
multiProcess = False
verbose = True
saveResult = True
# Let's try to use the basis gate of IBM Quantum Computers
allowedGates = [X, SqrtX, CNOT, Rz, Swap]
