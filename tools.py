import matplotlib.pyplot as plt
import pickle
import numpy as np
from datetime import datetime

#   Functions for handling and analyzing 
#   the population and logbook data

def problemName(pop, logbook, stateName):
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
  line1 = ax1.plot(gen, fit_mins, "b-", label=f"{fitness} Fitness")
  ax1.set_xlabel("Generation")
  ax1.set_ylabel("Fitness", color="b")
  for tl in ax1.get_yticklabels():
      tl.set_color("b")

  ax2 = ax1.twinx()
  line2 = ax2.plot(gen, size_avgs, "r-", label=f"{size} Size")
  ax2.set_ylabel("Size", color="r")
  for tl in ax2.get_yticklabels():
      tl.set_color("r")

  lns = line1 + line2
  labs = [l.get_label() for l in lns]
  ax1.legend(lns, labs, loc="center right")

  plt.show()

#   WIP
def plotPopFitSize(pop):
    return 0

def plotCircLengths(circs, circs2):
    sizes1 = np.array([circ.size() for circ in circs])
    max_size = sizes1.max()
    sizes2 = np.array([circ.size() for circ in circs2])
    if sizes2.max() > max_size:
        max_size = sizes2.max()
    plt.hist(sizes1, bins=max_size, range=(0,max_size), alpha=0.5)
    plt.hist(sizes2, bins=max_size, range=(0,max_size), alpha=0.5)
    plt.show()
