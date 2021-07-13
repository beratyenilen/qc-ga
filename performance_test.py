from constants import *
from scoop_test import main
from matplotlib import pyplot as plt
from deap import creator, base, tools
import numpy as np
import os
import pickle

def evaluateInd(individual, wanted, verbose=False):
    got = individual.simulateCircuit()
    error = 1 - np.absolute(np.vdot(wanted, got))**2
    individual.setCMW(error)
    if verbose:
        print("Wanted state is:", wanted)
        print("Produced state is", got)
        print("Error is:", error)
    if len(individual.circuit) > 0 and len(individual.circuit) < MAX_CIRCUIT_LENGTH:
        #return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
        return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
    else:
        return (error, 1.0)

def reordering(old):
    new = []
    new.append(old[len(old)-1])
    new.append(old[len(old)-2])
    new.append(old[len(old)-3])
    new.append(old[len(old)-4])
    new.append(old[len(old)-5])
    for i in range(0,len(old)-5):
        new.append(old[i])
    return new
        
gens = [10,20,30,40,50,60,80,100,150,200,250,300]

#for g in gens:
#    for i in range(100):
#        os.system('python3 scoop_test.py -g '+str(g)+' -i '+str(i+1))

avg_best_fidelities = []
best_length = []    # This is the length of circuit with the best fidelity. Not necessarily the shortest circuit
deviations = []


directory = f"performance_data/{numberOfQubits}QB/{POPSIZE}POP/"
i = 0
for gen in gens:
    length_sum = 0
    fidelity_sum = 0
    for j in range(100):
        stateName = f"{numberOfQubits}QB_state{j+1}"
        name = f"{i}-{gen}GEN-{stateName}.pop"

        f = open(directory+name, 'rb')
#        logbook = pickle.load(f)
        pop = pickle.load(f)
        f.close()
#        fit_mins = logbook.chapters["fitness"].select('min')
#        fidelity_sum += fit_mins[len(fit_mins) - 1] 
        f = open('states/'+str(numberOfQubits)+'_qubits/' + stateName, 'rb')
        desired_state = pickle.load(f)
        f.close()
        fidelity_sum += evaluateInd(pop[0], desired_state)[0]
        length_sum += len(pop[0].circuit)
        i+=1
    avg_best_fidelities.append(fidelity_sum/100)
    best_length.append(length_sum/100)

    # Calculating standard deviation
    i-=100
    std = 0
    for j in range(100):
        stateName = f"{numberOfQubits}QB_state{j+1}"
        name = f"{i}-{gen}GEN-{stateName}.pop"

        f = open(directory+name, 'rb')
#        logbook = pickle.load(f)
        pop = pickle.load(f)
        f.close()
#        fit_mins = logbook.chapters["fitness"].select('min')
#        fidelity_sum += fit_mins[len(fit_mins) - 1] 
        f = open('states/'+str(numberOfQubits)+'_qubits/' + stateName, 'rb')
        desired_state = pickle.load(f)
        f.close()
#        std += (evaluateInd(pop[0], desired_state)[0] - fidelity_sum/100) ** 2
        std += (len(pop[0].circuit) - length_sum/100) ** 2
        i+=1
    std = np.sqrt( std / 100 ) 
    deviations.append(std)


    
#gens = reordering(gens)
#avg_best_fidelities = reordering(avg_best_fidelities)
#best_length = reordering(best_length)
print(gens)
print(avg_best_fidelities)


fig, ax1 = plt.subplots()
plt.ylim(0,1)
line1 = ax1.plot(gens, avg_best_fidelities, "bo-", label="Fidelity")
ax1.set_xlabel("Generation")
ax1.set_ylabel("Fidelity", color="b")
for tl in ax1.get_yticklabels():
  tl.set_color("b")

ax2 = ax1.twinx()
line2 = ax2.plot(gens, best_length, "ro-", label="Size")
ax2.errorbar(gens, best_length, deviations, color="r")
ax2.set_ylabel("Size", color="r")
for tl in ax2.get_yticklabels():
  tl.set_color("r")

lns = line1 + line2
lns = line1
labs = [l.get_label() for l in lns]
#ax1.legend(lns, labs, loc="center right")

plt.title(f"{numberOfQubits} Qubits, POPSIZE={POPSIZE}")
plt.show()
