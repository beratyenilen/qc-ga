from json import tool
import matplotlib.pyplot as plt
import pickle
from scipy.special import gamma
import numpy as np
import copy
from datetime import datetime
from deap import creator, base, tools
from deap.tools.emo import sortNondominated
from projectq.ops import CNOT
from qiskit.providers.aer.noise import NoiseModel
from qiskit import transpile, Aer, execute
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import state_fidelity, DensityMatrix, Operator
from qiskit.circuit.library import Permutation
from qclib.state_preparation.baa_schmidt import initialize

from constants import *
from candidate import qasm2ls, Candidate
from new_evolution import *

 
#   Functions for handling and analyzing 
#   the population and logbook data

def problemName(pop, logbook, path, state_name):
    n = pop[0].numberOfQubits
    NGEN = len(logbook.select("gen")) 
    time = datetime.now()
    time_str = time.strftime("%d.%m.%y-%H:%M")
    f = open(path+time_str+"-"+str(len(pop))+"pop-"+str(NGEN)+"GEN-"+state_name+".pop", 'wb')

#   Save a population object and a logbook   
def save(pop, logbook, path, problemName):
    f = open(path+problemName+".pop", 'wb')
    pickle.dump(pop, f)
    f.close()
    f = open(path+problemName+".logbook", 'wb')
    pickle.dump(logbook, f)
    f.close()
    print('Saved!')

#
#   Load a population object and corresponding logbook.
#
#   Path contains the name of pop/logbook file 
#   WITHOUT .pop/.logbook extension
#
def load(path):
    f = open(path+".pop", 'rb')
    pop = pickle.load(f)
    f.close()
    f = open(path+".logbook", 'rb')
    logbook = pickle.load(f)
    f.close()
    return pop, logbook
    
def loadState(numberOfQubits, index):
    stateName = str(numberOfQubits)+"QB_state"+str(index)
    f = open('states/'+str(numberOfQubits)+'_qubits/' + stateName, 'rb')
    desired_state = pickle.load(f)
    f.close()
    return desired_state


#   Plot the fitness and size
def plotFitSize(logbook, fitness="min", size="avg"):
  """
    Values for fitness and size:
        "min" plots the minimum 
        "max" plots the maximum
        "avg" plots the average 
        "std" plots the standard deviation
  """
  gen = logbook.select("gen")
  fit_mins = logbook.chapters["fitness"].select(fitness)
  size_avgs = logbook.chapters["size"].select(size)

  fig, ax1 = plt.subplots()
  line1 = ax1.plot(gen, fit_mins, "b-", label=f"{fitness} Error")
  ax1.set_xlabel("Generation")
  ax1.set_ylabel("Error", color="b")
  for tl in ax1.get_yticklabels():
      tl.set_color("b")
  plt.ylim(0,1)

  ax2 = ax1.twinx()
  line2 = ax2.plot(gen, size_avgs, "r-", label=f"{size} Size")
  ax2.set_ylabel("Size", color="r")
  for tl in ax2.get_yticklabels():
      tl.set_color("r")

  lns = line1 + line2
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc="center right")


  plt.show()

def plotCircLengths(circs, circs2):
    sizes1 = np.array([circ.size() for circ in circs])
    max_size = sizes1.max()
    sizes2 = np.array([circ.size() for circ in circs2])
    if sizes2.max() > max_size:
        max_size = sizes2.max()
    plt.hist(sizes1, bins=max_size, range=(0,max_size), alpha=0.5)
    plt.show()

def plotLenFidScatter(pop):
    data = []
    for circ in pop:
        data.append([circ.toQiskitCircuit().size(), 1 - circ.fitness.values[0]])
    #plt.scatter(circ.toQiskitCircuit().size(), 1-evaluateInd(circ, desired_state)[0])
        
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, marker='.')
    plt.ylabel("Fidelity")
    plt.xlabel("Length")
    plt.ylim(0,1)
    #plt.title("Evaluation by length (1000 gen)")
    plt.show()

def fitfidScatter(pop, color=[0,0,0], all=True):
  ranks = sortNondominated(pop, len(pop), first_front_only=True)
  front = ranks[0]
  data = []
  for i in range(len(ranks[0]), len(pop)):
      pop[i].trim()
      circ = pop[i]
      data.append([circ.fitness.values[1], 1 - circ.fitness.values[0]])
  data = np.array(data)
  x = data[:, 0]
  y = data[:, 1]
  if (all):
    plt.scatter(x, y, color='b', marker='.')
  data = []
  data = np.array([[circ.fitness.values[1], 1 - circ.fitness.values[0]] for circ in front])
  x = data[:, 0]
  y = data[:, 1]
  plt.scatter(x, y, color=color, marker='.')
  plt.ylabel("Fidelity")
  plt.xlabel("Cost")
  plt.ylim(0,1)
#  plt.show()

def fitNoisefidScatter(pop, state_vector, fake_machine, noise_model, color=[0,0,0], all=True):
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    x = []
    y = []
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]

    if (all): front = pop
    
    for circ in front:
        x.append(circ.fitness.values[1])
        permutation = circ.getPermutationMatrix()
        permutation = np.linalg.inv(permutation)
        circ = circ.toQiskitCircuit()
        circ.measure_all()
        circ = transpile(circ,fake_machine,optimization_level=0)
        permutation2 = getPermutation(circ)
        perm_circ = Permutation(5, permutation2) # Creating a circuit for qubit mapping
        perm_unitary = Operator(perm_circ) # Matrix for the previous circuit

        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(perm_unitary @ permutation @ state_vector, density_matrix_noisy))
        

    #plt.scatter(x, y, color=color, marker='.')
    plt.ylabel("Fidelity")
    plt.xlabel("Cost")
    plt.ylim(0,1)
    return x, y

def plotCNOTSFidScatter(pop):
  ranks = sortNondominated(pop, len(pop), first_front_only=True)
  front = ranks[0]
  data = []
  for ind in front:
        circ = ind.circuit
        l = len(circ)
        if (l==0) :
            continue
        cnots = 0
        for gate in circ:
            if (gate[1]==CNOT): cnots+=1
        data.append([cnots, 1 - ind.fitness.values[0]])
  data = np.array(data)
  x = data[:, 0]
  y = data[:, 1]
  #plt.scatter(x, y, color=color, marker='.')
  plt.ylabel("Fidelity")
  plt.xlabel("CNOTS")
  plt.ylim(0,1)
  return x,y

def plotCNOTSNoiseFidScatter(pop, state_vector, fake_machine, noise_model, color=[0,0,0]):
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    x = []
    y = []
    for ind in front:
        circ = ind.circuit
        if (len(circ)==0) :
            continue
        cnots = 0
        for gate in circ:
            if (gate[1]==CNOT): cnots+=1
        permutation = ind.getPermutationMatrix()
        permutation = np.linalg.inv(permutation)
        circ = ind.toQiskitCircuit()
        circ.measure_all()
        circ = transpile(circ,fake_machine,optimization_level=0)
        permutation2 = getPermutation(circ)
        perm_circ = Permutation(5, permutation2) # Creating a circuit for qubit mapping
        perm_unitary = Operator(perm_circ) # Matrix for the previous circuit


        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(perm_unitary @ permutation @ state_vector, density_matrix_noisy))
        x.append(cnots)
    #plt.scatter(x, y, color=color, marker='.')
    plt.ylabel("Fidelity")
    plt.xlabel("CNOTS")
    plt.ylim(0,1)
    return x,y

def plotLenCNOTScatter(pop, color=[0,0,0]):
  ranks = sortNondominated(pop, len(pop), first_front_only=True)
  front = ranks[0]
  data = []
  for ind in front:
        circ = ind.circuit
        l = len(circ)
        if (l==0) :
            continue
        cnots = 0
        for gate in circ:
            if (gate[1]==CNOT): cnots+=1
        data.append([l,cnots])
  data = np.array(data)
  x = data[:, 0]
  y = data[:, 1]
  plt.scatter(x, y, color=color, marker='.')
  plt.ylabel("CNOTS")
  plt.xlabel("Length")

def paretoFront(pop, color=[0,0,0], all=True):
  ranks = sortNondominated(pop, len(pop), first_front_only=True)
  front = ranks[0]
  data = []
  for i in range(len(ranks[0]), len(pop)):
      pop[i].trim()
      circ = pop[i]
      data.append([circ.toQiskitCircuit().size(), 1 - circ.fitness.values[0]])
#   plt.scatter(circ.toQiskitCircuit().size(), 1-evaluateInd(circ, desired_state)[0])
      
  data = np.array(data)
  x = data[:, 0]
  y = data[:, 1]
  if (all):
    plt.scatter(x, y, color='b', marker='.')
  data = []
  data = np.array([[circ.toQiskitCircuit().size(), 1 - circ.fitness.values[0]] for circ in front])

#for i in range(0, len(front)):
#    pop[i].trim()
#    circ = pop[i]
#    data.append([circ.toQiskitCircuit().size(), 1 - circ.fitness.values[0]])
  x = data[:, 0]
  y = data[:, 1]
  plt.scatter(x, y, color=color, marker='.')
  plt.ylabel("Fidelity")
  plt.xlabel("Length")
  plt.ylim(0,1)
#  plt.show()

def theoreticalModel(n=5, p=0.01, l_lim=60, L=4):
    # p: Probability of error
    # L: Pairs of qubits connected by CNOT gates
    d = 2**n-n-1
    
    c1 = -(2*np.log(2)*n+np.log(L))/d
    c2 = -np.log(2)*n**2/d

    l = np.linspace(0,l_lim,1000)
    y = ((1-p)**l)*(1-np.exp(c1*l+c2))
    plt.plot(l,y)
    p = 0.005
    y = ((1-p)**l)*(1-np.exp(c1*l+c2))
    #plt.plot(l,y)
    y = ((1-p)**l)*(1-np.exp(c1*l+c2))
    #plt.plot(x,y)
    p = 0.0075
    y = ((1-p)**l)*(1-np.exp(c1*l+c2))
    #plt.plot(x,y)
#    y = ((1-p)**x)*(1-np.exp(c1*x))
#    plt.plot(x,y)

def paretoNoiseFids(pop, state_vector, fake_machine, noise_model, all=True, color='red'):
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    #backend = AerSimulator(method='density_matrix')

    x=[]
    y=[]
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    if (all): front = pop
    
    for circ in front:
        permutation = circ.getPermutationMatrix()
        permutation = np.linalg.inv(permutation)
        circ = circ.toQiskitCircuit()
        circ.measure_all()
        circ = transpile(circ,fake_machine,optimization_level=0)
        permutation2 = getPermutation(circ)
        perm_circ = Permutation(5, permutation2) # Creating a circuit for qubit mapping
        perm_unitary = Operator(perm_circ) # Matrix for the previous circuit

        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(perm_unitary @ permutation @ state_vector, density_matrix_noisy))
        x.append(circ.size())
    #plt.scatter(x,y, marker='.', color=color)
    plt.ylabel("Fidelity")
    plt.xlabel("Length")
    plt.ylim(0,1)
    #plt.show()
    return x,y

def saveQiskitCircuit(qc, name):
    path = "qiskit_circuits/"
    f = open(path+name, 'wb')
    pickle.dump(qc, f)
    f.close()

def loadQiskitCircuit(file):
    path = "qiskit_circuits/"
    f = open(path+file, 'rb')
    qc = pickle.load(f)
    f.close()
    return qc

def addMeasureGates(circ, n):
    circ.barrier()
    for i in range(n):
        circ.measure(i, i)
    return circ

def getPermutation(circ):
    #   Computing the permutations done by
    #   transpile function
    #ancilla = [0,1,2,3,4]
    perm = [0,1,2,3,4]
    for op, qubits, clbits in circ.data:
        if op.name == 'measure':
            a = perm.index(clbits[0].index)
            b = perm[qubits[0].index]
            #ancilla.remove(qubits[0].index)
            perm[qubits[0].index] = clbits[0].index
            perm[a] = b
    circ.remove_final_measurements()
    return perm

def get_perm_aug_vec(circs, desired_vector, n):
    n_iter = len(circs)
    pad_vectors = []    #   List of perm_aug_desired_vectors for every circ
    for i in range(n_iter):
        n_phys=5
        qubit_pattern = getPermutation(circs[i])
        print(qubit_pattern)
        circs[i].remove_final_measurements()
        circs[i].snapshot_density_matrix('final')


        aug_desired_vector = desired_vector

        for k in range(n_phys-n):
            aug_desired_vector = np.kron([1,0],aug_desired_vector)


        perm_circ = Permutation(n_phys, qubit_pattern) # Creating a circuit for qubit mapping
        perm_unitary = Operator(perm_circ) # Matrix for the previous circuit

        #perm_aug_desired_vector = perm_unitary.data @ aug_desired_vector
        perm_aug_desired_vector = aug_desired_vector
        pad_vectors.append(perm_aug_desired_vector)
    return pad_vectors


def LRSP_circs(state, toolbox):
#---------------------------------
    # define list of fidelity loss values to try out
    losses = list(np.linspace(0.0,1.0,80))
    pop = toolbox.population(n=1)
    # find the exact circuit
    circuit = initialize(state, max_fidelity_loss=0.0, strategy="brute_force", use_low_rank=True)
    circuit.measure_all()
    transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=3)

    # create a list of circuits with increasing fidelity loss
    circuits = [transpiled_circuit]

    for loss in losses:
        # find approximate initialization circuit with fidelity loss
        circuit = initialize(state, max_fidelity_loss=loss, strategy="brute_force", use_low_rank=True)
        circuit.measure_all()
        transpiled_circuit = transpile(circuit, basis_gates=basis_gates, optimization_level=3)

        #if transpiled_circuit.depth() < circuits[-1].depth():
        circuits.append(transpiled_circuit)
#---------------------------------

    #qiskit_circs, depths = genCircs(numberOfQubits, fake_machine, desiredState(), n_iter=1, measuregates=True)
    unaltered = []
    for i in range(len(circuits)):
        perm = getPermutation(circuits[i])
        circ = qasm2ls(circuits[i].qasm())
        pop[0].circuit = circ
        pop[0].permutation = perm
        pop[0].fitness.values = toolbox.evaluate(pop[0])
        unaltered.append(copy.deepcopy(pop[0]))
    return unaltered

    

def totalCNOTs(circ):
    cnots = 0
    for gate in circ:
        if (gate[1]==CNOT): cnots+=1
    return cnots

def avgCNOTs(pop):
    avg_cnots = 0
    total = 0
    for ind in pop:
        circ = ind.circuit
        l = len(circ)
        if (l==0) :
            continue
        cnots = 0
        for gate in circ:
            if (gate[1]==CNOT): cnots+=1
        avg_cnots += cnots/l
        total += 1
    avg_cnots = avg_cnots/total
    return avg_cnots


def evaluateIndcostt(desired_state):
    def _lambda(individual, verbose=False):
        """
        This function should take an individual,possibly an instance of Candidate class,
        and return a tuple where each element of the tuple is an objective.
        An example objective would be (error,circuitLen) where:
        error = |1 - < createdState | wantedState >
        circuitLen = len(individual.circuit) / MAX_CIRCUIT_LENGTH
        MAX_CIRCUIT_LENGTH is the expected circuit length for the problem.
        """
        got = individual.simulateCircuit()
        error = 1 - np.absolute(np.vdot(desired_state, got))**2
        individual.setCMW(error)
        cost = individual.evaluateCost()
        if verbose:
            print("Wanted state is:", desired_state)
            print("Produced state is", got)
            print("Error is:", error)
        return (error,cost)
    return _lambda

def initialize_toolbox(desired_state):
    toolbox = base.Toolbox()

    toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates, connectivity)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", crossoverInd, toolbox=toolbox)
    toolbox.register("mutate", mutateInd)
    toolbox.register("select", tools.selSPEA2)
    toolbox.register("selectAndEvolve", selectAndEvolve)
    toolbox.register("evaluate", evaluateIndcostt(desired_state))

    return toolbox