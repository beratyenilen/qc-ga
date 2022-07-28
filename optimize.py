from deap import creator, base, tools
from tools import *
from constants import *
from individual import Individual
from os.path import exists
from scipy.optimize import minimize
from matplotlib import pyplot as plt
import argparse

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-p", "--POPSIZE", help="Size of the population")
parser.add_argument("-g", "--NGEN", help="The number of generations")
parser.add_argument("-q", "--NQUBIT", help="The number of qubits")
parser.add_argument("-i", "--INDEX", help="Index of desired state")
parser.add_argument("-id", "--ID", help="ID of the saved file")

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
    load_file = True

# trying to minimize error and length !
fitnessWeights = (-1.0, -0.5)

# Create the type of the individual
creator.create("fitness_min", base.Fitness, weights=fitnessWeights)
creator.create("individual", Candidate, fitness=creator.fitness_min)


def total_cnots(circ):
    cnots = 0
    for gate in circ:
        if (gate[1] == CNOT):
            cnots += 1
    return cnots


def getRotationGates(individual):
    ''' 
    This function iterates over all the gates defined in the circuit and 
    returns a list of rotation gate indices.
    '''
    indices = []
    values = []
    for index in range(len(individual.circuit)):
        if individual.circuit[index][0] == "SG":
            indices.append(index)
            values.append(individual.circuit[index][3])
    return indices, values


def setParametersAndEvaluate(params, indices, individual):
    for i, p in zip(indices, params):
        # print(individual.circuit[i])
        individual.circuit[i] = (
            "SG", individual.circuit[i][1], individual.circuit[i][2], p)
        # print(individual.circuit[i])
    wanted = desired_state
    got = individual.simulateCircuit()
    return 1-np.absolute(np.vdot(wanted, got))**2


if __name__ == "__main__":

    for stateIndex in range(12, 101):
        unaltered_list = []
        stateName = str(numberOfQubits)+"QB_state"+str(stateIndex)
        state = load_state(5, stateIndex).data
        desired_state = state
        path = f"performance_data/5QB/400POP/500-30000GEN-5QB_state{stateIndex}"
        if (not exists(path+".pop")):
            # print(path)
            exit()
        pop, log = load(path)
        o_popfid = []
        o_poplen = []

        # print(rGates)
        # print(params)

        a = 0
        ranks = sortNondominated(pop, len(pop), first_front_only=True)
        front = ranks[0]
        for c in front:
            rGates, params = getRotationGates(c)
            if len(rGates) == 0:
                continue
            bounds = [(-2*np.pi, 2*np.pi) for _ in params]
            result = minimize(setParametersAndEvaluate, params,
                              args=(rGates, c), bounds=bounds)
            o_popfid.append(1-setParametersAndEvaluate(result.x, rGates, c))
            o_poplen.append(total_cnots(c.circuit))
            print(a)
            a += 1
#        plt.scatter(o_poplen, o_popfid, color='blue', marker='.')
#        plotCNOTSFidScatter(front)
#        plt.ylim(0,1)
#        plt.show()
#        paretoFront(pop, 'red')
#        plt.show()
        path = "optimizedfront_general/"
        problem_name = f"optimized-general{stateIndex}"
        f = open(path+problem_name+".pop", 'wb')
        pickle.dump(front, f)
        f.close()
