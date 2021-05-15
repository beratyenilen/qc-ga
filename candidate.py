import random
import copy
import numpy.random
import numpy as np
import projectq
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,CX,Rx,Ry,Rz,SqrtX
from projectq.ops import Measure,All,get_inverse,Swap,SwapGate
from math import pi
from qiskit import QuantumCircuit, transpile


class Candidate:
    """
    This class is a container for an individual in GA.
    """

    def __init__(self,numberOfQubits=2,allowedGates=[X,Rz,SqrtX,Swap,CNOT],connectivity="ALL",EMC=2.0,ESL=2.0,):
        self.numberOfQubits = numberOfQubits
        self.allowedGates = allowedGates
        self.connectivity = connectivity
        self.permutation = random.sample(range(self.numberOfQubits), numberOfQubits)
        # EMC stands for Expected Mutation Count
        self.EMC = EMC
        # ESL stands for Expected Sequence Length
        self.ESL = ESL
        self.circuit = self.generateRandomCircuit()
        self.CMW = 0.2
        self.optimized = False

    def getPermutationMatrix(self):
        """
        Args:
            perm: a list representing where each qubit is mapped to.
                perm[i] represents the physical qubit and i represents the virtual qubit.
                So [1,0,2] will be interpreted as (virtual->physical) 0->1, 1->0, 2->2
        Returns:
            2^N x 2^N numpy matrix representing the action of the permutation where
            N is the number of qubits.
        """
        Nexp = 2 ** self.numberOfQubits
        M = np.zeros((Nexp, Nexp))
        for frm in range(Nexp):
            toBin = list(bin(0)[2:].zfill(self.numberOfQubits))
            frmBin = list(bin(frm)[2:].zfill(self.numberOfQubits))
            toBin.reverse()
            frmBin.reverse()
            for p in range(self.numberOfQubits):
                toBin[p] = frmBin[self.permutation.index(p)]
            toBin.reverse()
            to = int("".join(toBin), 2)
            M[to][frm] = 1.0
        return M

    def __str__(self):
        output = "numberOfQubits: " + str(self.numberOfQubits)
        output += "\nConnectivity = " + str(self.connectivity)
        output += "\nQubit Mapping = " + str(self.permutation)
        output += "\nallowedGates: ["
        for i in range(len(self.allowedGates)):
            if self.allowedGates[i] == Rx:
                output += "Rx, "
            elif self.allowedGates[i] == Ry:
                output += "Ry, "
            elif self.allowedGates[i] == Rz:
                output += "Rz, "
            elif self.allowedGates[i] in [SwapGate, Swap]:
                output += "Swap, "
            elif self.allowedGates[i] in [SqrtX]:
                output += "SqrtX, "
            elif self.allowedGates[i] in [CNOT, CX]:
                output += "CX, "
            else:
                output += str(self.allowedGates[i]) + ", "
        output = output[:-2]
        output += "]\nEMC: " + str(self.EMC) + ", ESL: " + str(self.ESL) + "\n"
        output += self.printCircuit()
        output += "\ncircuitLength: " + str(len(self.circuit))
        return output

    def print(self):
        print(self)

    def printCircuit(self, verbose=True):
        output = "Qubit Mapping:" + str(self.permutation) + "\n"
        output += "Circuit: ["
        for i in range(len(self.circuit)):
            if self.circuit[i][0] == "SFG":
                output += (
                    "("
                    + str(self.circuit[i][1])
                    + ","
                    + str(self.circuit[i][2])
                    + "), "
                )
            elif self.circuit[i][0] == "TFG":
                output += (
                    "("
                    + str(self.circuit[i][1])
                    + ","
                    + str(self.circuit[i][2])
                    + ","
                    + str(self.circuit[i][3])
                    + "), "
                )
            elif self.circuit[i][0] == "SG":
                output += (
                    "("
                    + str(self.circuit[i][1](round(self.circuit[i][3], 3)))
                    + ","
                    + str(self.circuit[i][2])
                    + "), "
                )
        output = output[:-2]
        output += "]"
        return output

    def generateRandomCircuit(self, initialize=True):
        """
        Generates a random circuit with its length chosen from a geometric distr.
        with mean value self.ESL.
        Args:
          None
        Returns:
          A list of tuples representing a circuit.
        """
        if initialize:
            # If we are generating a random circuit while creating a new individual
            # we should choose the mean value of the geometric dist. to be 30.
            p = 1 / 30
        else:
            # If not, than we are basically using this in a mutation in which case
            # we should choose the mean value of the geometric dist. to be ESL.
            p = 1 / self.ESL
        # Produced circuit will have this length
        cirLength = numpy.random.geometric(p)
        producedCircuit = []
        for i in range(cirLength):
            # Choose a gate to add from allowedGates
            gate = random.choice(self.allowedGates)
            if gate in [CNOT, CX, Swap, SwapGate]:
                # if gate to add is CNOT we need to choose control and target indices
                if self.connectivity == "ALL":
                    control, target = random.sample(range(self.numberOfQubits), 2)
                else:
                    control, target = random.choice(self.connectivity)
                # TFG stands for Two Qubit Fixed Gate
                producedCircuit.append(("TFG", gate, control, target))
            elif gate in [H, X, Y, Z, T, Tdagger, S, Sdagger, SqrtX]:
                # choose the index to apply
                target = random.choice(range(self.numberOfQubits))
                # SFG stands for Single Qubit Fixed Gate
                producedCircuit.append(("SFG", gate, target))
            elif gate in [Rx, Ry, Rz]:
                # choose the index to apply the operator
                target = random.choice(range(self.numberOfQubits))
                # choose the rotation parameter between 0 and 2pi up to 3 significiant figures
                # !! We may change this significant figure things later on
                significantFigure = 2
                parameter = round(pi * random.uniform(0, 2), significantFigure)
                producedCircuit.append(("SG", gate, target, parameter))
            else:
                print("WHAT ARE YOU DOING HERE!!")
                print("GATE IS:", gate)
                # At a later time we will add the following types:
                # 	  - TG, Two Qubit Gate with Parameters [CRz] they will be
                # 	represented as ("TG", gate, control, target, parameter)

        # Produced circuit will be a list of tuples in the following form:
        # [(type, gate, index), (type, gate, control, target), ...]
        return producedCircuit

    def simulateCircuit(self):
        """
        Simulates the action of self.circuit and return the resulting state.
        Args:
          None --> maybe we can add a return type in here later.
        Returns:
          A numpy.darray representing the resulting state.
        """
        # There are various backend in projectq. We will be using Simulator backend
        # in order to have access to the full wavefunction.
        sim = projectq.backends.Simulator()
        # We need to define the MainEngine, which will be our compiler.
        # Let us remove engine_list from the equation. (I didn't understand what it is yet)
        eng = projectq.MainEngine(backend=sim, engine_list=[])
        qureg = eng.allocate_qureg(self.numberOfQubits)

        # Now we can start applying the gates as proposed in self.circuit
        for i in range(len(self.circuit)):  # I am not sure about this way of checking circuit length, this may create problems if I am not careful
            if self.circuit[i][0] == "SFG":
                gate = self.circuit[i][1]
                target = self.circuit[i][2]
                gate | qureg[target]
            elif self.circuit[i][0] == "TFG":
                gate = self.circuit[i][1]
                control = self.circuit[i][2]
                target = self.circuit[i][3]
                gate | (qureg[control], qureg[target])
            elif self.circuit[i][0] == "SG":
                gate = self.circuit[i][1]
                target = self.circuit[i][2]
                parameter = self.circuit[i][3]
                gate(parameter) | qureg[target]
            else:
                print("WRONG BRANCH IN simulateCircuit")
        # This sends all the added gates to the compiler to be optimized.
        # Without this, we may have problems.
        eng.flush()
        # Let us get a deepcopy of the our resulting state.
        mapping, wavefunction = copy.deepcopy(eng.backend.cheat())

        # I am not sure if we need mapping at any point of our calculations due
        # to the fact that we'll only be using Simulator backend, but I need to do
        # some documentation reading to understand what's going on.

        # I will simply return the wavefunction due to the fact that the mapping is
        # probably almost the same ?? I NEED TO VERIFY THIIS !!!!

        # If we don't measure our circuit in the end we get an exception and since
        # we deepcopied our state to wavefunction this shouldn't create a problem.
        All(Measure) | qureg

        # Multiply it with the permutation
        return self.getPermutationMatrix() @ wavefunction

    def drawCircuit(self, form="qiskit"):
        """
        Args:
            if type='projectq' this function draws the circuit using
                projecq.backends.CircuitDrawerMatplotlib
            if type='qiskit' this function draws the circuit using
                 qiskit.QuantumCircuit.draw()
        Returns:
          figure object
        """
        if form == "projectq":
            # Set the backend and initalize the registers.
            drawBackend = projectq.backends.CircuitDrawerMatplotlib()
            eng = projectq.MainEngine(drawBackend, engine_list=[])
            qureg = eng.allocate_qureg(self.numberOfQubits)

            # Constructing the circuit.
            for i in range(len(self.circuit)):
                if self.circuit[i][0] == "SFG":
                    gate = self.circuit[i][1]
                    target = self.circuit[i][2]
                    gate | qureg[target]
                elif self.circuit[i][0] == "TFG":
                    gate = self.circuit[i][1]
                    control = self.circuit[i][2]
                    target = self.circuit[i][3]
                    gate | (qureg[control], qureg[target])
                elif self.circuit[i][0] == "SG":
                    gate = self.circuit[i][1]
                    target = self.circuit[i][2]
                    parameter = self.circuit[i][3]
                    gate(parameter) | qureg[target]
                else:
                    print("WRONG BRANCH IN drawCircuit")
            # This sends all the added gates to the compiler to be optimized.
            # Without this, we may have problems.
            eng.flush()

            # In order to plot, we need to input labels for out qubits and a drawing order.
            # I will simply name qubits as 0,1 for the moment, later on we can change this.
            qubitLabels = {}
            # I will choose the topmost qubit to be the zeroth qubit.
            drawingOrder = {}
            for i in range(self.numberOfQubits):
                # qubitLabels[i] = i
                qubitLabels[i] = self.permutation[i]
                drawingOrder[i] = self.numberOfQubits - 1 - i

            # drawBackend.draw(qubitLabels, drawingOrder)[0].show()
            return drawBackend.draw(qubitLabels, drawingOrder)[0]

        else:
            return self.toQiskitCircuit().draw("mpl")

    def toQiskitCircuit(self):
        """
        Returns: qiskit.QuantumCircuit object of the circuit of the Candidate
        """
        qc = QuantumCircuit(self.numberOfQubits)
        for op in self.circuit:
            if op[0] == "TFG":
                # can be CNOT,CX,Swap,SwapGate
                if op[1] in [CX, CNOT]:
                    qc.cx(op[2], op[3])
                elif op[1] in [Swap, SwapGate]:
                    qc.swap(op[2], op[3])
                else:
                    print("Problem in toQiskitCircuit:", op[1])

            elif op[0] == "SFG":
                # can be H,X,Y,Z,T,T^d,S,S^d,sqrtX,sqrtXdagger
                if op[1] == H:
                    qc.h(op[2])
                elif op[1] == X:
                    qc.x(op[2])
                elif op[1] == Y:
                    qc.y(op[2])
                elif op[1] == Z:
                    qc.z(op[2])
                elif op[1] == T:
                    qc.t(op[2])
                elif op[1] == Tdagger:
                    qc.tdg(op[2])
                elif op[1] == S:
                    qc.s(op[2])
                elif op[1] == Sdagger:
                    qc.sdg(op[2])
                elif op[1] == SqrtX:
                    qc.sx(op[2])
                elif op[1] == get_inverse(SqrtX):
                    qc.sxdg(op[2])
                else:
                    print("Problem in toQiskitCircuit:", op[1])

            elif op[0] == "SG":
                # can be Rx,Ry,Rz
                if op[1] == Rx:
                    qc.rx(op[3], op[2])
                elif op[1] == Ry:
                    qc.ry(op[3], op[2])
                elif op[1] == Rz:
                    qc.rz(op[3], op[2])
                else:
                    print("Problem in toQiskitCircuit:", op[1])

        return qc

    def optimize(self, optimization_level=2):
        """
        Optimizes self.circuit using qiskit.transpile(optimization_level=2).
        """
        basis = []
        for gate in self.allowedGates:
            if gate == H:
                basis.append("h")
            elif gate == X:
                basis.append("x")
            elif gate == Y:
                basis.append("y")
            elif gate == Z:
                basis.append("z")
            elif gate == T:
                basis.append("t")
            elif gate == Tdagger:
                basis.append("tdg")
            elif gate == S:
                basis.append("s")
            elif gate == Sdagger:
                basis.append("sdg")
            elif gate == SqrtX:
                basis.append("sx")
            elif gate == get_inverse(SqrtX):
                basis.append("sxdg")
            elif gate in [CX, CNOT]:
                basis.append("cx")
            elif gate in [Swap, SwapGate]:
                basis.append("swap")
            elif gate == Rx:
                basis.append("rx")
            elif gate == Ry:
                basis.append("ry")
            elif gate == Rz:
                basis.append("rz")

        if self.connectivity == "ALL":
            qc = transpile(
                self.toQiskitCircuit(),
                optimization_level=optimization_level,
                basis_gates=basis,
                layout_method="trivial",
            )
        else:
            qc = transpile(
                self.toQiskitCircuit(),
                optimization_level=optimization_level,
                basis_gates=basis,
                layout_method="trivial",
            )
        self.circuit = qasm2ls(qc.qasm())
        self.optimized = True

    def evaluateCost(self):
        """
        Evaluates the cost of the self.circuit.
        Each SWAP gate is 30, CNOT is 10, and all the single qubit gates
        are 1 points worth.
        """
        cost = 0
        for op in self.circuit:
            if op[0] in ["SG", "SFG"]:
                cost += 1
            elif op[1] in [SwapGate, Swap]:
                cost += 30
            elif op[1] in [CNOT, CX]:
                cost += 10
        return cost

    def setCMW(self,error):
        """
            Sets CMW (Continuous Mutation Width) to error / 5. 
        """
        self.CMW = error / 5
        
    ###################### Mutations from here on ############################

    def permutationMutation(self, verbose=False):
        '''
        Randomly changes the permutation of the individual.
        '''
        self.permutation = random.sample(range(self.numberOfQubits), self.numberOfQubits)


    def discreteUniformMutation(self, verbose=False):
        """
        This function iterates over all the gates defined in the circuit and
        randomly changes target and/or control qubits with probability EMC / circuitLength.
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuitLength = len(self.circuit)
        if circuitLength == 0:
            mutationProb = 0
        else:
            mutationProb = self.EMC / circuitLength
        # I don't know if we really need this part
        if mutationProb >= 1.0:
            mutationProb = 0.5

        # We will loop over all the gates
        for i in range(circuitLength):
            if random.random() < mutationProb:
                self.discreteMutation(i, verbose=verbose)

    def sequenceInsertion(self, verbose=False):
        """
        This function generates a random circuit with circuit length given by choosing
        a value from a geometric distribution with mean value ESL, and it is inserted
        to a random point in self.circuit.
        """
        circuitToInsert = self.generateRandomCircuit(initialize=False)
        oldCircuitLength = len(self.circuit)
        if oldCircuitLength == 0:
            insertionIndex = 0
        else:
            insertionIndex = random.choice(range(oldCircuitLength))
        self.circuit[insertionIndex:] = circuitToInsert + self.circuit[insertionIndex:]

    def sequenceAndInverseInsertion(self, verbose=False):
        """
        This function generates a random circuit with circuit length given by choosing
        a value from a geometric distribution with mean value ESL, it is inserted to a
        random point in self.circuit and its inverse is inserted to another point.
        """
        circuitToInsert = self.generateRandomCircuit(initialize=False)
        # MAYBE CONNECTIVITY IS NOT REFLECTIVE ?
        inverseCircuit = getInverseCircuit(circuitToInsert, verbose)
        oldCircuitLength = len(self.circuit)
        if oldCircuitLength >= 2:
            index1, index2 = random.sample(range(oldCircuitLength), 2)
            if index1 > index2:
                index2, index1 = index1, index2
        else:
            index1, index2 = 0, 1
        newCircuit = (
            self.circuit[:index1]
            + circuitToInsert
            + self.circuit[index1:index2]
            + inverseCircuit
            + self.circuit[index2:]
        )
        self.circuit = newCircuit

    def discreteMutation(self, index, verbose=False):
        """
        This function applies a discrete mutation to the circuit element at index.
        Discrete mutation means that the control and/or target qubits are randomly changed.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1
        if self.circuit[index][0] == "SFG":
            # This means we have a single qubit fixed gate
            newTarget = random.choice(range(self.numberOfQubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], newTarget)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                newControl, newTarget = random.sample(range(self.numberOfQubits), 2)
            else:
                newControl, newTarget = random.choice(self.connectivity)
            self.circuit[index] = ("TFG", self.circuit[index][1], newControl, newTarget)
        elif self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            newTarget = random.choice(range(self.numberOfQubits))
            self.circuit[index] = (
                "SG",
                self.circuit[index][1],
                newTarget,
                self.circuit[index][3],
            )
        else:
            print("WRONG BRANCH IN discreteMutation")

    def continuousMutation(self, index, verbose=False):
        """
        This function applies a continuous mutation to the circuit element at index.
        Continuous mutation means that if the gate has a parameter, its parameter its
        changed randomly, if not a discreteMutation is applied.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1

        if self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            newParameter = float(self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
            self.circuit[index] = ("SG",self.circuit[index][1],self.circuit[index][2],newParameter)
        elif self.circuit[index][0] == "SFG":
            # This means we have a single qubit/two qubit fixed gate and we need to
            # apply a discreteMutation.
            newTarget = random.choice(range(self.numberOfQubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], newTarget)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                newControl, newTarget = random.sample(range(self.numberOfQubits), 2)
            else:
                newControl, newTarget = random.choice(self.connectivity)
            self.circuit[index] = ("TFG", self.circuit[index][1], newControl, newTarget)
        else:
            print("WRONG BRANCH IN continuousMutation")
    
    def parameterMutatıon(self):
        ''' 
        This function iterates over all the gates defined in the circuit and 
        randomly adjusts the parameter of the rotation gates.
        '''
        if len(self.circuit) == 0:
            return

        mutationProb = self.EMC / len(self.circuit)
        for index in range(len(self.circuit)):
            if random.random() < mutationProb:
                if self.circuit[index][0] == "SG":
                    # This means we have a single rotation gate
                    newParameter = float(self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
                    self.circuit[index] = ("SG",self.circuit[index][1],self.circuit[index][2],newParameter)
                '''
                # Maybe we can also apply a discrete mutation we'll see.
                elif self.circuit[index][0] == "SFG":
                    # This means we have a single qubit/two qubit fixed gate and we need to
                    # apply a discreteMutation.
                    newTarget = random.choice(range(self.numberOfQubits))
                    self.circuit[index] = ("SFG", self.circuit[index][1], newTarget)
                elif self.circuit[index][0] == "TFG":
                    # This means we have two qubit fixed gate
                    if self.connectivity == "ALL":
                        newControl, newTarget = random.sample(range(self.numberOfQubits), 2)
                    else:
                        newControl, newTarget = random.choice(self.connectivity)
                    self.circuit[index] = ("TFG", self.circuit[index][1], newControl, newTarget)
                else:
                    print("WRONG BRANCH IN continuousMutation")
                '''
        
    def continuousUniformMutation(self, verbose=False):
        """
        This function iterates over all the gates defined in the circuit and
        randomly changes the parameter if possible, if not target and/or control qubits
        with probability EMC / circuitLength.
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuitLength = len(self.circuit)
        if circuitLength == 0:
            mutationProb = 0
        else:
            mutationProb = self.EMC / circuitLength
        # I don't know if we really need this part
        if mutationProb >= 1.0:
            mutationProb = 0.5

        # We will loop over all the gates
        for i in range(circuitLength):
            if random.random() < mutationProb:
                self.continuousMutation(i, verbose=verbose)

    def insertMutateInvert(self, verbose=False):
        """
        This function performs a discrete mutation on a single gate, then places a
        randomly selected gate immediately before it and its inverse immediately
        after it.
        """
        # index to apply discrete mutation
        if len(self.circuit) == 0:
            index = 0
        else:
            index = random.choice(range(len(self.circuit)))

        # Discrete Mutation
        self.discreteMutation(index)

        # Generate the circuit to insert and its inverse
        circuitToInsert = self.generateRandomCircuit(initialize=False)
        while len(circuitToInsert) == 0:
            circuitToInsert = self.generateRandomCircuit(initialize=False)
        circuitToInsert = [circuitToInsert[0]]
        inverseCircuit = getInverseCircuit(circuitToInsert)
        if index >= len(self.circuit):
            # This probably happens only when index = 0 and length of the circuit = 0
            if index == 0:
                newCircuit = circuitToInsert + inverseCircuit
            else:
                print("\n\nIT SHOULD NEVER ENTER HEREE!!!\n\n")
        else:
            newCircuit = (
                self.circuit[:index]
                + circuitToInsert
                + [self.circuit[index]]
                + inverseCircuit
                + self.circuit[(index + 1) :]
            )
        self.circuit = newCircuit

    def swapQubits(self, verbose=False):
        """
        This function swaps two randomly selected qubits.
        """
        qubit1, qubit2 = random.sample(range(self.numberOfQubits), 2)
        if verbose:
            print("\nqubit1:", qubit1, " qubit2:", qubit2)
            print("Before swapQubits:")
            self.printCircuit()

        for i in range(len(self.circuit)):
            if self.circuit[i][0] == "SFG":
                if self.circuit[i][2] == qubit1:
                    self.circuit[i] = self.circuit[i][0:2] + (qubit2,)
                elif self.circuit[i][2] == qubit2:
                    self.circuit[i] = self.circuit[i][0:2] + (qubit1,)

            elif self.circuit[i][0] == "TFG":
                if self.circuit[i][2] == qubit1 and self.circuit[i][3] == qubit2:
                    self.circuit[i] = self.circuit[i][0:2] + (qubit2, qubit1)

                elif self.circuit[i][2] == qubit2 and self.circuit[i][3] == qubit1:
                    self.circuit[i] = self.circuit[i][0:2] + (qubit1, qubit2)

                elif self.circuit[i][2] == qubit1:
                    self.circuit[i] = (
                        self.circuit[i][0:2] + (qubit2,) + self.circuit[i][3:]
                    )

                elif self.circuit[i][2] == qubit2:
                    self.circuit[i] = (
                        self.circuit[i][0:2] + (qubit1,) + self.circuit[i][3:]
                    )

                elif self.circuit[i][3] == qubit1:
                    self.circuit[i] = self.circuit[i][0:3] + (qubit2,)

                elif self.circuit[i][3] == qubit2:
                    self.circuit[i] = self.circuit[i][0:3] + (qubit1,)

            elif self.circuit[i][0] == "SG":
                if self.circuit[i][2] == qubit1:
                    self.circuit[i] = (
                        self.circuit[i][0:2] + (qubit2,) + (self.circuit[i][3],)
                    )
                elif self.circuit[i][2] == qubit2:
                    self.circuit[i] = (
                        self.circuit[i][0:2] + (qubit1,) + (self.circuit[i][3],)
                    )

        if verbose:
            print("After swapQubits:")
            self.printCircuit()

    def sequenceDeletion(self, verbose=False):
        """
        This function deletes a randomly selected interval of the circuit.
        """
        if len(self.circuit) < 2:
            return

        circuitLength = len(self.circuit)
        index = random.choice(range(circuitLength))
        if verbose:
            print("\nindex:", index)
            print("Before sequenceDeletion:")
            self.printCircuit()
        # If this is the case, we'll simply remove the last element
        if index == (circuitLength - 1):
            self.circuit = self.circuit[:-1]
        else:
            sequenceLength = numpy.random.geometric(p=(1 / self.ESL))
            if verbose:
                print("sequenceLength:", sequenceLength)
            if (index + sequenceLength) >= circuitLength:
                self.circuit = self.circuit[: (-circuitLength + index)]
            else:
                self.circuit = (
                    self.circuit[:index] + self.circuit[(index + sequenceLength) :]
                )

        if verbose:
            print("After sequenceDeletion:")
            self.printCircuit()

    def sequenceReplacement(self, verbose=False):
        """
        This function first applies sequenceDeletion, then applies a sequenceInsertion.
        """
        if verbose:
            print("\nBefore sequenceDeletion:")
            self.printCircuit()

        self.sequenceDeletion()

        if verbose:
            print("After sequenceDeletion and before sequenceInsertion:")
            self.printCircuit()

        self.sequenceInsertion()

        if verbose:
            print("After sequenceInsertion:")
            self.printCircuit()

    def sequenceSwap(self, verbose=False):
        """
        This function randomly chooses two parts of the circuit and swaps them.
        """
        if len(self.circuit) < 4:
            return

        indices = random.sample(range(len(self.circuit)), 4)
        indices.sort()
        i1, i2, i3, i4 = indices[0], indices[1], indices[2], indices[3]
        if verbose:
            print(
                "\nBefore sequenceSwap index1:",
                i1,
                "index2:",
                i2,
                "index3:",
                i3,
                "index4:",
                i4,
            )
            self.printCircuit()

        self.circuit = (
            self.circuit[0:i1]
            + self.circuit[i3:i4]
            + self.circuit[i2:i3]
            + self.circuit[i1:i2]
            + self.circuit[i4:]
        )

        if verbose:
            print("After sequeceSwap:")
            self.printCircuit()

    def sequenceScramble(self, verbose=False):
        """
        This function randomly chooses an index and chooses a length from a geometric
        dist. w/ mean value ESL, and permutes the gates in that part of the circuit.
        """
        circuitLength = len(self.circuit)
        if circuitLength < 2:
            index1 = 0
        else:
            index1 = random.choice(range(circuitLength - 1))

        sequenceLength = numpy.random.geometric(p=(1 / self.ESL))
        if (index1 + sequenceLength) >= circuitLength:
            index2 = circuitLength - 1
        else:
            index2 = index1 + sequenceLength

        toShuffle = self.circuit[index1:index2]
        random.shuffle(toShuffle)

        if verbose:
            print("\nBefore sequenceScramble w/ index1:", index1, "index2:", index2)
            self.printCircuit()

        self.circuit = self.circuit[:index1] + toShuffle + self.circuit[index2:]

        if verbose:
            print("After sequenceScramble:")
            self.printCircuit()

    def moveGate(self, verbose=False):
        """
        This function randomly moves a gate from one point to another point.
        """
        circuitLength = len(self.circuit)
        if circuitLength < 2:
            return
        oldIndex, newIndex = random.sample(range(circuitLength), 2)
        if verbose:
            print("\nBefore moveGate w/ oldIndex", oldIndex, "newIndex", newIndex)
            self.printCircuit()

        temp = self.circuit.pop(oldIndex)
        self.circuit.insert(newIndex, temp)

        if verbose:
            print("After moveGate:")
            self.printCircuit()

    def crossOver(parent1, parent2, child, verbose=False):
        """
        This function gets two parent solutions, creates an empty child, randomly
        picks the number of gates to be selected from each parent and selects that
        number of gates from the first parent, and discards that many from the
        second parent. Repeats this until parent solutions are exhausted.
        Args:
          Parent solutions.
        Returns:
          (child1, child2).
        """
        parent1Circuit = parent1.circuit[:]
        parent2Circuit = parent2.circuit[:]
        p1 = p2 = 1.0

        if len(parent1Circuit) != 0:
            p1 = parent1.EMC / len(parent1.circuit)
        if (p1 <= 0) or (p1 > 1):
            p1 = 1.0

        if len(parent2Circuit) != 0:
            p2 = parent2.EMC / len(parent2.circuit)
        if (p2 <= 0) or (p2 > 1):
            p2 = 1.0

        if verbose:
            print("\nCrossover between parent1:")
            parent1.printCircuit()
            print("And parent2:")
            parent2.printCircuit()

        child.circuit = []
        turn = 1
        while len(parent1Circuit) or len(parent2Circuit):
            if verbose:
                print("turn:", turn)
                print("Before crossover:")
                child.printCircuit()
            if turn == 1:
                numberOfGatesToSelect = numpy.random.geometric(p1)
                child.circuit += parent1Circuit[:numberOfGatesToSelect]
                turn = 2
            else:
                numberOfGatesToSelect = numpy.random.geometric(p2)
                child.circuit += parent2Circuit[:numberOfGatesToSelect]
                turn = 1
            if verbose:
                print("numberOfGatesToSelect:", numberOfGatesToSelect)
                print("After crossover")
                child.printCircuit()
            parent1Circuit = parent1Circuit[numberOfGatesToSelect:]
            parent2Circuit = parent2Circuit[numberOfGatesToSelect:]


def printCircuit(circuit, verbose=True):
    output = "Circuit: ["
    for i in range(len(circuit)):
        if circuit[i][0] == "SFG":
            output += "(" + str(circuit[i][1]) + "," + str(circuit[i][2]) + "), "
        elif circuit[i][0] == "TFG":
            output += (
                "("
                + str(circuit[i][1])
                + ","
                + str(circuit[i][2])
                + ","
                + str(circuit[i][3])
                + "), "
            )
        elif circuit[i][0] == "SG":
            output += (
                "("
                + str(circuit[i][1](round(circuit[i][3], 3)))
                + ","
                + str(circuit[i][2])
                + "), "
            )
    output = output[:-2]
    output += "]"
    if verbose:
        print(output)
    return output


def qasm2ls(qasmstr):
    """
        This helper function takes in qasmstr representing a quantum circuit
        and then transforms it into our list representation of the circuit.
    Returns: list
    """
    operations = qasmstr.split("\n")[3:]
    oplist = []
    for op in operations:
        opl = op.split(" ")
        if opl[0] in ["x", "h", "y", "z", "t", "tdg", "s", "sdg", "sx", "sxdg"]:
            t = int(opl[1][2:-2])
            if opl[0] == "x":
                oplist.append(("SFG", X, t))
            elif opl[0] == "h":
                oplist.append(("SFG", H, t))
            elif opl[0] == "y":
                oplist.append(("SFG", Y, t))
            elif opl[0] == "z":
                oplist.append(("SFG", Z, t))
            elif opl[0] == "t":
                oplist.append(("SFG", T, t))
            elif opl[0] == "tdg":
                oplist.append(("SFG", Tdagger, t))
            elif opl[0] == "s":
                oplist.append(("SFG", S, t))
            elif opl[0] == "sdg":
                oplist.append(("SFG", Sdagger, t))
            elif opl[0] == "sx":
                oplist.append(("SFG", SqrtX, t))
            elif opl[0] == "sxdg":
                oplist.append(("SFG", get_inverse(SqrtX), t))

        elif opl[0] in ["cx", "swap"]:
            co, ta = opl[1].split(",")
            co = int(co[2:-1])
            ta = int(ta[2:-2])
            if opl[0] == "cx":
                oplist.append(("TFG", CX, co, ta))
            else:
                oplist.append(("TFG", Swap, co, ta))

        elif opl[0][:2] in ["rx", "ry", "rz"]:
            t = int(opl[1][2:-2])
            if opl[0][3:-1] in ["pi", "-pi"]:
                p = pi
            elif "pi" in opl[0][3:-1] and "*" in opl[0][3:-1]:
                p = float(opl[0][3:-1].split("*")[0]) * pi
            elif "pi" in opl[0][3:-1] and "/" in opl[0][3:-1]:
                p = pi / float(opl[0][3:-1].split("/")[1])
            else:
                p = float(opl[0][3:-1])

            if opl[0][:2] == "rx":
                oplist.append(("SG", Rx, t, p))
            elif opl[0][:2] == "ry":
                oplist.append(("SG", Ry, t, p))
            elif opl[0][:2] == "rz":
                oplist.append(("SG", Rz, t, p))

        elif opl[0] == "":
            continue
        else:
            print("error in qasm2ls:", opl)

    return oplist


def getInverseCircuit(circuit, verbose=False):
    """
    This function takes a circuit and returns a circuit which is the inverse circuit.
    """
    if len(circuit) == 0:
        return []

    reversedCircuit = circuit[::-1]
    for i in range(len(reversedCircuit)):
        if reversedCircuit[i][1] in [H, X, Y, Z, CX, Swap, SwapGate]:
            continue
        elif reversedCircuit[i][1] == S:
            reversedCircuit[i] = ("SFG", Sdagger, reversedCircuit[i][2])
        elif reversedCircuit[i][1] == Sdagger:
            reversedCircuit[i] = ("SFG", S, reversedCircuit[i][2])
        elif reversedCircuit[i][1] == T:
            reversedCircuit[i] = ("SFG", Tdagger, reversedCircuit[i][2])
        elif reversedCircuit[i][1] == Tdagger:
            reversedCircuit[i] = ("SFG", T, reversedCircuit[i][2])
        elif reversedCircuit[i][1] in [Rx, Ry, Rz]:
            reversedCircuit[i] = (
                "SG",
                reversedCircuit[i][1],
                reversedCircuit[i][2],
                round(2 * pi - reversedCircuit[i][3], 3),
            )
        elif reversedCircuit[i][1] in [SqrtX]:
            reversedCircuit[i] = ("SFG", get_inverse(SqrtX), reversedCircuit[i][2])
        elif reversedCircuit[i][1] in [get_inverse(SqrtX)]:
            reversedCircuit[i] = ("SFG", SqrtX, reversedCircuit[i][2])
        else:
            print("\nWRONG BRANCH IN getInverseCircuit\n")

    if verbose:
        print("Circuit to invert:")
        printCircuit(circuit)
        print("Inverse circuit:")
        printCircuit(reversedCircuit)

    return reversedCircuit


def testDiscreteUniformMutation(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.discreteUniformMutation(verbose)


def testSequenceInsertion(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.sequenceInsertion(verbose)


def testSequenceAndInverseInsertion(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.sequenceAndInverseInsertion(verbose)


def testInsertMutateInvert(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.insertMutateInvert(verbose)


def testSwapQubits(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.swapQubits(verbose)


def testSequenceDeletion(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.sequenceDeletion(verbose)


def testSequenceReplacement(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.sequenceReplacement(verbose)


def testSequenceSwao(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.sequenceSwap(verbose)


def testSequenceScramble(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.sequenceScramble(verbose)


def testMoveGate(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.moveGate(verbose)


def testContinuousUniformMutation(candidate, trials=10, verbose=False):
    for i in range(trials):
        candidate.continuousUniformMutation(verbose)


def testOptimize(c):
    c.drawCircuit().savefig("./c10.png")
    print("Before optimization:")
    print(c.toQiskitCircuit().count_ops())
    wfs = []
    wfs.append(c.simulateCircuit())
    for optlvl in [0,1, 2, 3]:
        c2 = Candidate(c.numberOfQubits)
        c2.circuit = c.circuit
        c2.optimize(optlvl)
        c2.drawCircuit().savefig("./c" + str(optlvl) + ".png")
        print("Optimization level:", optlvl)
        print(c2.toQiskitCircuit().count_ops())
        wfs.append(c2.simulateCircuit())
    return wfs


allowedGates = [H, X, Y, Z, CX, Rx, Ry, Rz, S, Sdagger, T, Tdagger, Swap, SqrtX]
