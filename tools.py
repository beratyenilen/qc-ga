import matplotlib.pyplot as plt
import pickle
import argparse
import numpy as np
from scipy.special import gamma
from datetime import datetime
from deap.tools.emo import sortNondominated
from qiskit import transpile, Aer, execute
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer import AerSimulator
from qiskit.quantum_info import state_fidelity, DensityMatrix

#   Functions for handling and analyzing 
#   the population and logbook data

def problemName(pop, logbook, path, state_name):
    n = pop[0].numberOfQubits
    NGEN = len(logbook.select("gen")) 
    time = datetime.now()
    time_str = time.strftime("%d.%m.%y-%H:%M")
#    ID = time.strftime("%d%m%y%H%M%S")+str(len(pop))+str(NGEN)+str(n)   #This needs improving
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

#   Load allowed gateset, fitness and seed from a file
def getSetup(path):
    return 0

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


def plotLenFidScatter(directory, problemName, numberOfQubits, stateName, evaluateInd, POPSIZE):
    name = problemName+".pop"
    f = open('states/'+str(numberOfQubits)+'_qubits/' + stateName, 'rb')
    desired_state = pickle.load(f)
    f.close()

    data = []
    c=[]
    for i in range(POPSIZE):
        c.append(i)
        f = open(directory+name, 'rb')
        pop = pickle.load(f)
        f.close()
        pop[i].trim()
        circ = pop[i]
        data.append([circ.toQiskitCircuit().size(), 1 - evaluateInd(circ)[0]])
#    plt.scatter(circ.toQiskitCircuit().size(), 1-evaluateInd(circ, desired_state)[0])
        
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, c=c)
    plt.ylabel("Fidelity")
    plt.xlabel("Length")
#plt.xlim(0,400)
    plt.ylim(0,1)
#    plt.title("Evaluation by length (1000 gen)")
    plt.colorbar()
    plt.show()

def paretoFront(pop):
  n = 5
# Pairs of qubits connected by CNOT gates
  L = 8
  d = 2**n-n-1  
  c1 = -(2*np.log(2)*n+np.log(L))/d
  c2 = -np.log(2)*n**2/d
  x = np.linspace(0,300,3000)
  y = 1-np.exp(c1*x+c2)-0.2
#  plt.plot(x,y)
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
  plt.scatter(x, y, color='b')
  data = []
  data = np.array([[circ.toQiskitCircuit().size(), 1 - circ.fitness.values[0]] for circ in front])

#  for i in range(0, len(front)):
#      pop[i].trim()
#      circ = pop[i]
#      data.append([circ.toQiskitCircuit().size(), 1 - circ.fitness.values[0]])
  x = data[:, 0]
  y = data[:, 1]
  plt.scatter(x, y, color='r')
  plt.ylabel("Fidelity")
  plt.xlabel("Length")
  plt.ylim(0,1)
  plt.show()

def paretoNoiseFids(pop, state_vector, fake_machine):
    # Circuit length
    l = 1
    # Probability of error
    p = 0.001
    # Number of qubits
    n = 5
    # Pairs of qubits connected by CNOT gates
    L = 8
    d = 2**n-n-1

    Cn = (2*np.sqrt(np.pi))**(1-n)*gamma(d/2+1)/gamma(2**n)
    
    c1 = -(2*np.log(2)*n+np.log(L))/d
    c2 = -np.log(2)*n**2/d

    x = np.linspace(0,300,3000)
    y = ((1-p)**x)*(1-np.exp(c1*x+c2))
    plt.plot(x,y)
    p = 0.005
    y = ((1-p)**x)*(1-np.exp(c1*x+c2))
    plt.plot(x,y)
    p = 0.0075
    y = ((1-p)**x)*(1-np.exp(c1*x+c2))
    plt.plot(x,y)
#    y = ((1-p)**x)*(1-np.exp(c1*x))
#    plt.plot(x,y)

    noise_model = NoiseModel.from_backend(fake_machine)
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    x=[]
    y=[]
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    front = pop
    print(front[0].toQiskitCircuit().size())
    density_matrix = DensityMatrix.from_instruction(front[0].toQiskitCircuit())
    for circ in front:
        circ = circ.toQiskitCircuit()
        circ = transpile(circ,fake_machine,optimization_level=0)
        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(state_vector, density_matrix_noisy))
        x.append(circ.size())
    plt.plot(x,y,'o')
    plt.show()

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

if __name__ == "__main__":
    from constants import *
    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-f", "--FILE", help = "Path to the file without the final '.pop' or '.logbook' extension")
    parser.add_argument("-p", "--POPSIZE", help = "Size of the population")
    parser.add_argument("-g", "--NGEN", help = "The number of generations")
    parser.add_argument("-q", "--NQUBIT", help = "The number of qubits")
    parser.add_argument("-i", "--INDEX", help = "Index of desired state")
    parser.add_argument("-id", "--ID", help = "ID of the saved file")

    # Read arguments from command line
    args = parser.parse_args()

    if args.POPSIZE:
        POPSIZE = int(args.POPSIZE)
    if args.NGEN:
        NGEN = int(args.NGEN)
    if args.NQUBIT:
        numberOfQubits = int(args.NQUBIT)
    if args.INDEX:
        stateIndex = int(args.INDEX)
    if args.ID:
        ID = int(args.ID)

    #FILE_PATH = 'performance_data/'+str(numberOfQubits)+'QB/'+str(POPSIZE)+'POP/'+str(ID)+'-'+str(NGEN)+'GEN-'+str(numberOfQubits)+'QB_state'+str(stateIndex)
    FILE_PATH = 'saved/0-2000GEN-5QB_state33'
    if args.FILE:
        FILE_PATH = args.FILE
    pop, logbook = load(FILE_PATH)
    paretoNoiseFids(pop, fake_machine)
    
