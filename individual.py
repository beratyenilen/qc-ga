import random
import copy
import numpy.random
import numpy as np
import projectq
from projectq.ops import H, X, Y, Z, T, Tdagger, S, Sdagger, CNOT, CX, Rx, Ry, Rz, SqrtX, Measure, All, get_inverse, Swap, SwapGate
from math import pi
from qiskit import QuantumCircuit, transpile, QuantumRegister, ClassicalRegister

from tools import qasm2ls, string_of_projectq


class Individual:
    """This class is a container for an individual in GA.
    """

    def __init__(self, number_of_qubits, allowed_gates, connectivity="ALL", EMC=2.0, ESL=2.0,):
        self.number_of_qubits = number_of_qubits
        self.allowed_gates = allowed_gates
        self.connectivity = connectivity
        self.permutation = random.sample(
            range(number_of_qubits), number_of_qubits)
        # TODO why use initialism rather than full name for the variables?
        # EMC stands for Expected Mutation Count
        self.EMC = EMC
        # ESL stands for Expected Sequence Length
        self.ESL = ESL
        self.circuit = self.generate_random_circuit()
        self.CMW = 0.2
        self.optimized = False

    def get_permutation_matrix(self):
        """
        Args:
            perm: a list representing where each qubit is mapped to.
                perm[i] represents the physical qubit and i represents the virtual qubit.
                So [1,0,2] will be interpreted as (virtual->physical) 0->1, 1->0, 2->2
        Returns:
            2^N x 2^N numpy matrix representing the action of the permutation where
            N is the number of qubits.
        """
        Nexp = 2 ** self.number_of_qubits
        M = np.zeros((Nexp, Nexp))
        for frm in range(Nexp):
            to_bin = list(bin(0)[2:].zfill(self.number_of_qubits))
            frm_bin = list(bin(frm)[2:].zfill(self.number_of_qubits))
            to_bin.reverse()
            frm_bin.reverse()
            for p in range(self.number_of_qubits):
                to_bin[p] = frm_bin[self.permutation.index(p)]
            to_bin.reverse()
            to = int("".join(to_bin), 2)
            M[to][frm] = 1.0
        return M

    def __str__(self):
        output = "number_of_qubits: " + str(self.number_of_qubits)
        output += "\nConnectivity = " + str(self.connectivity)
        output += "\nQubit Mapping = " + str(self.permutation)
        output += "\nallowedGates: ["
        for i in range(len(self.allowed_gates)):
            if self.allowed_gates[i] == Rx:
                output += "Rx, "
            elif self.allowed_gates[i] == Ry:
                output += "Ry, "
            elif self.allowed_gates[i] == Rz:
                output += "Rz, "
            elif self.allowed_gates[i] in [SwapGate, Swap]:
                output += "Swap, "
            elif self.allowed_gates[i] in [SqrtX]:
                output += "SqrtX, "
            elif self.allowed_gates[i] in [CNOT, CX]:
                output += "CX, "
            else:
                output += str(self.allowed_gates[i]) + ", "
        output = output[:-2]
        output += "]\nEMC: " + str(self.EMC) + ", ESL: " + str(self.ESL) + "\n"
        output += self.print_circuit()
        output += "\ncircuitLength: " + str(len(self.circuit))
        return output

    def print(self):
        print(self)

    def print_circuit(self):
        output = "Qubit Mapping:" + str(self.permutation) + "\n"
        output += "Circuit: ["
        for operator in self.circuit:
            if operator[0] == "SFG":
                output += (
                    "("
                    + str(operator[1])
                    + ","
                    + str(operator[2])
                    + "), "
                )
            elif operator[0] == "TFG":
                output += (
                    "("
                    + str(operator[1])
                    + ","
                    + str(operator[2])
                    + ","
                    + str(operator[3])
                    + "), "
                )
            elif operator[0] == "SG":
                output += (
                    "("
                    + str(operator[1](round(operator[3], 3)))
                    + ","
                    + str(operator[2])
                    + "), "
                )
        output = output[:-2]
        output += "]"
        return output

    def generate_random_circuit(self, initialize=True):
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
        cir_length = numpy.random.geometric(p)
        produced_circuit = []
        for _ in range(cir_length):
            # Choose a gate to add from allowed_gates
            gate = random.choice(self.allowed_gates)
            if gate in [CNOT, CX, Swap, SwapGate]:
                # if gate to add is CNOT we need to choose control and target indices
                if self.connectivity == "ALL":
                    control, target = random.sample(
                        range(self.number_of_qubits), 2)
                else:
                    control, target = random.choice(self.connectivity)
                # TFG stands for Two Qubit Fixed Gate
                produced_circuit.append(("TFG", gate, control, target))
            elif gate in [H, X, Y, Z, T, Tdagger, S, Sdagger, SqrtX]:
                # choose the index to apply
                target = random.choice(range(self.number_of_qubits))
                # SFG stands for Single Qubit Fixed Gate
                produced_circuit.append(("SFG", gate, target))
            elif gate in [Rx, Ry, Rz]:
                # choose the index to apply the operator
                target = random.choice(range(self.number_of_qubits))
                # choose the rotation parameter between 0 and 2pi up to 3 significiant figures
                # !! We may change this significant figure things later on
                significant_figure = 2
                parameter = round(pi * random.uniform(0, 2),
                                  significant_figure)
                produced_circuit.append(("SG", gate, target, parameter))
            else:
                print("WHAT ARE YOU DOING HERE!!")
                print("GATE IS:", gate)
                # At a later time we will add the following types:
                # 	  - TG, Two Qubit Gate with Parameters [CRz] they will be
                # 	represented as ("TG", gate, control, target, parameter)

        # Produced circuit will be a list of tuples in the following form:
        # [(type, gate, index), (type, gate, control, target), ...]
        return produced_circuit

    def simulate_circuit(self):
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
        qureg = eng.allocate_qureg(self.number_of_qubits)

        # Now we can start applying the gates as proposed in self.circuit
        # I am not sure about this way of checking circuit length, this may create problems if I am not careful
        for operator in self.circuit:
            if operator[0] == "SFG":
                gate = operator[1]
                target = operator[2]
                gate | qureg[target]
            elif operator[0] == "TFG":
                gate = operator[1]
                control = operator[2]
                target = operator[3]
                gate | (qureg[control], qureg[target])
            elif operator[0] == "SG":
                gate = operator[1]
                target = operator[2]
                parameter = operator[3]
                gate(parameter) | qureg[target]
            else:
                print("WRONG BRANCH IN simulate_circuit")
        # This sends all the added gates to the compiler to be optimized.
        # Without this, we may have problems.
        eng.flush()
        # Let us get a deepcopy of the our resulting state.
        _, wavefunction = copy.deepcopy(eng.backend.cheat())

        # I am not sure if we need mapping at any point of our calculations due
        # to the fact that we'll only be using Simulator backend, but I need to do
        # some documentation reading to understand what's going on.

        # I will simply return the wavefunction due to the fact that the mapping is
        # probably almost the same ?? I NEED TO VERIFY THIIS !!!!

        # If we don't measure our circuit in the end we get an exception and since
        # we deepcopied our state to wavefunction this shouldn't create a problem.
        All(Measure) | qureg

        # Multiply it with the permutation
        return self.get_permutation_matrix() @ wavefunction

    def draw_circuit(self, form="qiskit"):
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
            qureg = eng.allocate_qureg(self.number_of_qubits)

            # Constructing the circuit.
            for operator in self.circuit:
                if operator[0] == "SFG":
                    gate = operator[1]
                    target = operator[2]
                    gate | qureg[target]
                elif operator[0] == "TFG":
                    gate = operator[1]
                    control = operator[2]
                    target = operator[3]
                    gate | (qureg[control], qureg[target])
                elif operator[0] == "SG":
                    gate = operator[1]
                    target = operator[2]
                    parameter = operator[3]
                    gate(parameter) | qureg[target]
                else:
                    print("WRONG BRANCH IN draw_circuit")
            # This sends all the added gates to the compiler to be optimized.
            # Without this, we may have problems.
            eng.flush()

            # In order to plot, we need to input labels for out qubits and a drawing order.
            # I will simply name qubits as 0,1 for the moment, later on we can change this.
            qubit_labels = {}
            # I will choose the topmost qubit to be the zeroth qubit.
            drawingOrder = {}
            for i in range(self.number_of_qubits):
                # qubit_labels[i] = i
                qubit_labels[i] = self.permutation[i]
                drawingOrder[i] = self.number_of_qubits - 1 - i

            # drawBackend.draw(qubit_labels, drawingOrder)[0].show()
            return drawBackend.draw(qubit_labels, drawingOrder)[0]

        else:
            return self.to_qiskit_circuit().draw("mpl")

    def to_qiskit_circuit(self):
        """
        Returns: qiskit.QuantumCircuit object of the circuit of the Candidate
        """
        qr = QuantumRegister(self.number_of_qubits)
        cr = ClassicalRegister(self.number_of_qubits)
        qc = QuantumCircuit(qr, cr)
        for op in self.circuit:
            if op[0] == "TFG":
                # can be CNOT,CX,Swap,SwapGate
                if op[1] in [CX, CNOT]:
                    qc.cx(op[2], op[3])
                elif op[1] in [Swap, SwapGate]:
                    qc.swap(op[2], op[3])
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

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
                    print("Problem in to_qiskit_circuit:", op[1])

            elif op[0] == "SG":
                # can be Rx,Ry,Rz
                if op[1] == Rx:
                    qc.rx(op[3], op[2])
                elif op[1] == Ry:
                    qc.ry(op[3], op[2])
                elif op[1] == Rz:
                    qc.rz(op[3], op[2])
                else:
                    print("Problem in to_qiskit_circuit:", op[1])

        return qc

    # FIXME Unused
    def optimize(self, optimization_level=2):
        """
        Optimizes self.circuit using qiskit.transpile(optimization_level=2).
        """

        basis = [string_of_projectq(gate) for gate in self.allowed_gates]

        if self.connectivity == "ALL":
            qc = transpile(
                self.to_qiskit_circuit(),
                optimization_level=optimization_level,
                basis_gates=basis,
                layout_method="trivial",
            )
        else:
            qc = transpile(
                self.to_qiskit_circuit(),
                optimization_level=optimization_level,
                basis_gates=basis,
                layout_method="trivial",
            )
        self.circuit = qasm2ls(qc.qasm())
        self.optimized = True

    def clean(self):
        """
        Optimizes self.circuit by removing redundant gates
        """
        finished = False
        while not finished:
            finished = True
            i = 0
            while i < len(self.circuit) - 1:
                gate = self.circuit[i]

                if gate[1] == SqrtX:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit[i] = ("SFG", X, gate[2])
                            finished = False
                            break
                        elif self.circuit[j][1] == get_inverse(SqrtX) and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == Rz:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            parameter = (self.circuit[j][3] + gate[3]) % (pi*2)
                            self.circuit.pop(j)
                            self.circuit[i] = ("SG", Rz, gate[2], parameter)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == X:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][2] == gate[2] or self.circuit[j][3] == gate[2]:
                                break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        j += 1
                elif gate[1] == CX:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2] and self.circuit[j][3] == gate[3]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        elif self.circuit[j][2] == gate[3]:
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][3] == gate[2] or self.circuit[j][3] == gate[3]:
                                break
                        j += 1
                elif gate[1] == Swap:
                    j = i+1
                    while j < len(self.circuit):
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[2] and self.circuit[j][3] == gate[3]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        if self.circuit[j][1] == gate[1] and self.circuit[j][2] == gate[3] and self.circuit[j][3] == gate[2]:
                            self.circuit.pop(j)
                            self.circuit.pop(i)
                            finished = False
                            break
                        elif self.circuit[j][2] == gate[2]:
                            break
                        elif self.circuit[j][2] == gate[3]:
                            break
                        elif self.circuit[j][0] == "TFG":
                            if self.circuit[j][3] == gate[2] or self.circuit[j][3] == gate[3]:
                                break
                        j += 1

                i += 1

    def evaluate_cost(self):
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
                cost += 30  # FIXME magic numbers -> constants.py
            elif op[1] in [CNOT, CX]:
                cost += 10
        return cost

    def setCMW(self, error):
        """
            Sets CMW (Continuous Mutation Width) to error / 5. 
        """
        self.CMW = error / 5  # FIXME magic number

    ###################### Mutations from here on ############################

    def permutation_mutation(self):
        '''
        Randomly changes the permutation of the individual.
        '''
        self.permutation = random.sample(
            range(self.number_of_qubits), self.number_of_qubits)

    def discrete_uniform_mutation(self):
        """
        This function iterates over all the gates defined in the circuit and
        randomly changes target and/or control qubits with probability EMC / circuit_length.
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuit_length = len(self.circuit)
        if circuit_length == 0:
            mutation_prob = 0
        else:
            mutation_prob = self.EMC / circuit_length
        # I don't know if we really need this part
        if mutation_prob >= 1.0:
            mutation_prob = 0.5

        # We will loop over all the gates
        for i in range(circuit_length):
            if random.random() < mutation_prob:
                self.discrete_mutation(i)

    def sequence_insertion(self):
        """
        This function generates a random circuit with circuit length given by choosing
        a value from a geometric distribution with mean value ESL, and it is inserted
        to a random point in self.circuit.
        """
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        old_circuit_length = len(self.circuit)
        if old_circuit_length == 0:
            insertion_index = 0
        else:
            insertion_index = random.choice(range(old_circuit_length))
        self.circuit[insertion_index:] = circuit_to_insert + \
            self.circuit[insertion_index:]

    def sequence_and_inverse_insertion(self):
        """
        This function generates a random circuit with circuit length given by choosing
        a value from a geometric distribution with mean value ESL, it is inserted to a
        random point in self.circuit and its inverse is inserted to another point.
        """
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        # MAYBE CONNECTIVITY IS NOT REFLECTIVE ?
        inverse_circuit = get_inverse_circuit(circuit_to_insert)
        old_circuit_length = len(self.circuit)
        if old_circuit_length >= 2:
            index1, index2 = random.sample(range(old_circuit_length), 2)
            if index1 > index2:
                index2, index1 = index1, index2
        else:
            index1, index2 = 0, 1
        new_circuit = (
            self.circuit[:index1]
            + circuit_to_insert
            + self.circuit[index1:index2]
            + inverse_circuit
            + self.circuit[index2:]
        )
        self.circuit = new_circuit

    def discrete_mutation(self, index):
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
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], new_target)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                new_control, new_target = random.sample(
                    range(self.number_of_qubits), 2)
            else:
                new_control, new_target = random.choice(self.connectivity)
            self.circuit[index] = (
                "TFG", self.circuit[index][1], new_control, new_target)
        elif self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = (
                "SG",
                self.circuit[index][1],
                new_target,
                self.circuit[index][3],
            )
        else:
            print("WRONG BRANCH IN discrete_mutation")

    def continuous_mutation(self, index):
        """
        This function applies a continuous mutation to the circuit element at index.
        Continuous mutation means that if the gate has a parameter, its parameter its
        changed randomly, if not a discrete_mutation is applied.
        """
        if len(self.circuit) == 0:
            return
        while index >= len(self.circuit):
            index -= 1

        if self.circuit[index][0] == "SG":
            # This means we have a single rotation gate
            newParameter = float(
                self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
            self.circuit[index] = (
                "SG", self.circuit[index][1], self.circuit[index][2], newParameter)
        elif self.circuit[index][0] == "SFG":
            # This means we have a single qubit/two qubit fixed gate and we need to
            # apply a discrete_mutation.
            new_target = random.choice(range(self.number_of_qubits))
            self.circuit[index] = ("SFG", self.circuit[index][1], new_target)
        elif self.circuit[index][0] == "TFG":
            # This means we have two qubit fixed gate
            if self.connectivity == "ALL":
                new_control, new_target = random.sample(
                    range(self.number_of_qubits), 2)
            else:
                new_control, new_target = random.choice(self.connectivity)
            self.circuit[index] = (
                "TFG", self.circuit[index][1], new_control, new_target)
        else:
            print("WRONG BRANCH IN continuous_mutation")

    def parameter_mutation(self):
        ''' 
        This function iterates over all the gates defined in the circuit and 
        randomly adjusts the parameter of the rotation gates.
        '''
        if len(self.circuit) == 0:
            return

        mutation_prob = self.EMC / len(self.circuit)
        for index in range(len(self.circuit)):
            if random.random() < mutation_prob:
                if self.circuit[index][0] == "SG":
                    # This means we have a single rotation gate
                    newParameter = float(
                        self.circuit[index][-1]) + numpy.random.normal(scale=self.CMW)
                    newParameter = newParameter % (2*pi)
                    self.circuit[index] = (
                        "SG", self.circuit[index][1], self.circuit[index][2], newParameter)

    def continuous_uniform_mutation(self):
        """
        This function iterates over all the gates defined in the circuit and
        randomly changes the parameter if possible, if not target and/or control qubits
        with probability EMC / circuit_length.
        Args:
          None
        Returns:
          None -> should I return sth ? maybe self?
        """
        circuit_length = len(self.circuit)
        if circuit_length == 0:
            mutation_prob = 0
        else:
            mutation_prob = self.EMC / circuit_length
        # I don't know if we really need this part
        if mutation_prob >= 1.0:
            mutation_prob = 0.5

        # We will loop over all the gates
        for i in range(circuit_length):
            if random.random() < mutation_prob:
                self.continuous_mutation(i)

    def insert_mutate_invert(self):
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
        self.discrete_mutation(index)

        # Generate the circuit to insert and its inverse
        circuit_to_insert = self.generate_random_circuit(initialize=False)
        while len(circuit_to_insert) == 0:
            circuit_to_insert = self.generate_random_circuit(initialize=False)
        circuit_to_insert = [circuit_to_insert[0]]
        inverse_circuit = get_inverse_circuit(circuit_to_insert)
        if index >= len(self.circuit):
            # This probably happens only when index = 0 and length of the circuit = 0
            if index == 0:
                new_circuit = circuit_to_insert + inverse_circuit
            else:
                print("\n\nIT SHOULD NEVER ENTER HEREE!!!\n\n")
        else:
            new_circuit = (
                self.circuit[:index]
                + circuit_to_insert
                + [self.circuit[index]]
                + inverse_circuit
                + self.circuit[(index + 1):]
            )
        self.circuit = new_circuit

    def swap_qubits(self):
        """
        This function swaps two randomly selected qubits.
        """
        qubit1, qubit2 = random.sample(range(self.number_of_qubits), 2)

        for operator in self.circuit:
            if operator[0] == "SFG":
                if operator[2] == qubit1:
                    operator = operator[0:2] + (qubit2,)
                elif operator[2] == qubit2:
                    operator = operator[0:2] + (qubit1,)

            elif operator[0] == "TFG":
                if operator[2] == qubit1 and operator[3] == qubit2:
                    operator = operator[0:2] + (qubit2, qubit1)

                elif operator[2] == qubit2 and operator[3] == qubit1:
                    operator = operator[0:2] + (qubit1, qubit2)

                elif operator[2] == qubit1:
                    operator = (
                        operator[0:2] + (qubit2,) + operator[3:]
                    )

                elif operator[2] == qubit2:
                    operator = (
                        operator[0:2] + (qubit1,) + operator[3:]
                    )

                elif operator[3] == qubit1:
                    operator = operator[0:3] + (qubit2,)

                elif operator[3] == qubit2:
                    operator = operator[0:3] + (qubit1,)

            elif operator[0] == "SG":
                if operator[2] == qubit1:
                    operator = (
                        operator[0:2] +
                        (qubit2,) + (operator[3],)
                    )
                elif operator[2] == qubit2:
                    operator = (
                        operator[0:2] +
                        (qubit1,) + (operator[3],)
                    )

    def sequence_deletion(self):
        """
        This function deletes a randomly selected interval of the circuit.
        """
        if len(self.circuit) < 2:
            return

        circuit_length = len(self.circuit)
        index = random.choice(range(circuit_length))
        # If this is the case, we'll simply remove the last element
        if index == (circuit_length - 1):
            self.circuit = self.circuit[:-1]
        else:
            sequence_length = numpy.random.geometric(p=(1 / self.ESL))
            if (index + sequence_length) >= circuit_length:
                self.circuit = self.circuit[: (-circuit_length + index)]
            else:
                self.circuit = (
                    self.circuit[:index] +
                    self.circuit[(index + sequence_length):]
                )

    def sequence_replacement(self):
        """
        This function first applies sequence_deletion, then applies a sequence_insertion.
        """
        self.sequence_deletion()
        self.sequence_insertion()

    def sequence_swap(self):
        """
        This function randomly chooses two parts of the circuit and swaps them.
        """
        if len(self.circuit) < 4:
            return

        indices = random.sample(range(len(self.circuit)), 4)
        indices.sort()
        i1, i2, i3, i4 = indices[0], indices[1], indices[2], indices[3]

        self.circuit = (
            self.circuit[0:i1]
            + self.circuit[i3:i4]
            + self.circuit[i2:i3]
            + self.circuit[i1:i2]
            + self.circuit[i4:]
        )

    def sequence_scramble(self):
        """
        This function randomly chooses an index and chooses a length from a geometric
        dist. w/ mean value ESL, and permutes the gates in that part of the circuit.
        """
        circuit_length = len(self.circuit)
        if circuit_length < 2:
            index1 = 0
        else:
            index1 = random.choice(range(circuit_length - 1))

        sequence_length = numpy.random.geometric(p=(1 / self.ESL))
        if (index1 + sequence_length) >= circuit_length:
            index2 = circuit_length - 1
        else:
            index2 = index1 + sequence_length

        toShuffle = self.circuit[index1:index2]
        random.shuffle(toShuffle)

        self.circuit = self.circuit[:index1] + \
            toShuffle + self.circuit[index2:]

    def move_gate(self):
        """
        This function randomly moves a gate from one point to another point.
        """
        circuit_length = len(self.circuit)
        if circuit_length < 2:
            return
        old_index, new_index = random.sample(range(circuit_length), 2)

        temp = self.circuit.pop(old_index)
        self.circuit.insert(new_index, temp)

    def cross_over(self, parent2, toolbox):
        """This function gets two parent solutions, creates an empty child, randomly
        picks the number of gates to be selected from each parent and selects that
        number of gates from the first parent, and discards that many from the
        second parent. Repeats this until parent solutions are exhausted.
        """
        self_circuit = self.circuit[:]
        parent2_circuit = parent2.circuit[:]
        p1 = p2 = 1.0

        if len(self_circuit) != 0:
            p1 = self.EMC / len(self.circuit)
        if (p1 <= 0) or (p1 > 1):
            p1 = 1.0

        if len(parent2_circuit) != 0:
            p2 = parent2.EMC / len(parent2.circuit)
        if (p2 <= 0) or (p2 > 1):
            p2 = 1.0

        child = toolbox.individual()
        child.circuit = []
        turn = 1
        while len(self_circuit) or len(parent2_circuit):
            if turn == 1:
                number_of_gates_to_select = numpy.random.geometric(p1)
                child.circuit += self_circuit[:number_of_gates_to_select]
                turn = 2
            else:
                number_of_gates_to_select = numpy.random.geometric(p2)
                child.circuit += parent2_circuit[:number_of_gates_to_select]
                turn = 1
            self_circuit = self_circuit[number_of_gates_to_select:]
            parent2_circuit = parent2_circuit[number_of_gates_to_select:]
        return child


def print_circuit(circuit):
    output = "Circuit: ["
    for i in range(len(circuit)):
        if circuit[i][0] == "SFG":
            output += "(" + str(circuit[i][1]) + \
                "," + str(circuit[i][2]) + "), "
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
    return output


def get_inverse_circuit(circuit):
    """
    This function takes a circuit and returns a circuit which is the inverse circuit.
    """
    if len(circuit) == 0:
        return []

    reversed_circuit = circuit[::-1]
    for gate in reversed_circuit:
        if gate[1] in [H, X, Y, Z, CX, Swap, SwapGate]:
            continue
        elif gate[1] == S:
            gate = ("SFG", Sdagger, gate[2])
        elif gate[1] == Sdagger:
            gate = ("SFG", S, gate[2])
        elif gate[1] == T:
            gate = ("SFG", Tdagger, gate[2])
        elif gate[1] == Tdagger:
            gate = ("SFG", T, gate[2])
        elif gate[1] in [Rx, Ry, Rz]:
            gate = (
                "SG",
                gate[1],
                gate[2],
                round(2 * pi - gate[3], 3),
            )
        elif gate[1] in [SqrtX]:
            gate = ("SFG", get_inverse(
                SqrtX), gate[2])
        elif gate[1] in [get_inverse(SqrtX)]:
            gate = ("SFG", SqrtX, gate[2])
        else:
            print("\nWRONG BRANCH IN get_inverse_circuit\n")

    return reversed_circuit


def testDiscreteUniformMutation(candidate, trials=10):
    for i in range(trials):
        candidate.discreteUniformMutation()


def testSequenceInsertion(candidate, trials=10):
    for i in range(trials):
        candidate.sequence_insertion()


def testSequenceAndInverseInsertion(candidate, trials=10):
    for i in range(trials):
        candidate.sequence_and_inverse_insertion()


def testInsertMutateInvert(candidate, trials=10):
    for i in range(trials):
        candidate.insert_mutate_invert()


def testSwapQubits(candidate, trials=10):
    for i in range(trials):
        candidate.swap_qubits()


def testSequenceDeletion(candidate, trials=10):
    for i in range(trials):
        candidate.sequence_deletion()


def testSequenceReplacement(candidate, trials=10):
    for i in range(trials):
        candidate.sequence_replacement()


def testSequenceSwao(candidate, trials=10):
    for i in range(trials):
        candidate.sequence_swap()


def testSequenceScramble(candidate, trials=10):
    for i in range(trials):
        candidate.sequence_scramble()


def testMoveGate(candidate, trials=10):
    for i in range(trials):
        candidate.move_gate()


def testContinuousUniformMutation(candidate, trials=10):
    for i in range(trials):
        candidate.continuous_uniform_mutation()


def testOptimize(c):
    c.draw_circuit().savefig("./c10.png")
    print("Before optimization:")
    print(c.to_qiskit_circuit().count_ops())
    wfs = []
    wfs.append(c.simulate_circuit())
    for optlvl in [0, 1, 2, 3]:
        c2 = Individual(c.number_of_qubits)
        c2.circuit = c.circuit
        c2.optimize(optlvl)
        c2.draw_circuit().savefig("./c" + str(optlvl) + ".png")
        print("Optimization level:", optlvl)
        print(c2.to_qiskit_circuit().count_ops())
        wfs.append(c2.simulate_circuit())
    return wfs
