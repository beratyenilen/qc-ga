# Importing the necessary modules.
import projectq
from projectq.ops import H,X,Y,Z,T,Tdagger,S,Sdagger,CNOT,Measure,All,Rx,Ry,Rz,SqrtX,Swap
import numpy as np
import copy
from constants import *
from deap import creator, base, tools
from candidate import Candidate
from constants import *
from evolution import crossoverInd, mutateInd, selectAndEvolve, geneticAlgorithm
import pickle

# Import additional modules you want to use in here

# Let's say our state is:
# c0*|00> + c1*|01> + c2*|10> + c3*|11>
c0 = 0.3
c1 = 0.5
c2 = 0.66
c3 = 0


def desiredState():
    """
    This function returns the state vector of the desiredState as list where
    ith element is the ith coefficient of the state vector.
    """
    # wf = [c0,c1,c2,c3]
    wf = [
        0.15913149 + 0.05285494j,
        0.16971038 + 0.16935005j,
        0.20387004 + 0.07045581j,
        0.17458786 + 0.01685555j,
        0.09591196 + 0.1715848j,
        0.07813584 + 0.12933371j,
        0.14077403 + 0.04833421j,
        0.1864469 + 0.02210824j,

        0.04930783 + 0.17694084j,
        0.03305475 + 0.0548612j,
        0.15951635 + 0.02474114j,
        0.13083445 + 0.05724182j,
        0.1878547 + 0.03316511j,
        0.1711776 + 0.05024366j,
        0.03407228 + 0.12793345j,
        0.14235411 + 0.14044561j,

        0.04584613 + 0.02082665j,
        0.13775167 + 0.12803463j,
        0.17285302 + 0.02778957j,
        0.05822403 + 0.01662237j,
        0.05963336 + 0.16299819j,
        0.15122561 + 0.141271j,
        0.06105595 + 0.09488902j,
        0.01724461 + 0.13417011j,
        
        0.18846858 + 0.1717289j,
        0.04372904 + 0.20581626j,
        0.07840571 + 0.1703873j,
        0.09906241 + 0.06995841j,
        0.06082297 + 0.06730944j,
        0.21166858 + 0.04769224j,
        0.19487817 + 0.21000818j,
        0.17386463 + 0.072214j,
    ]
    return wf


def evaluateIndcostt(individual, verbose=False):
    """
    This function should take an individual,possibly an instance of Candidate class,
    and return a tuple where each element of the tuple is an objective.
    An example objective would be (error,circuitLen) where:
      error = |1 - < createdState | wantedState >
      circuitLen = len(individual.circuit) / MAX_CIRCUIT_LENGTH
      MAX_CIRCUIT_LENGTH is the expected circuit length for the problem.
    """
    wanted = desiredState()
    got = individual.simulateCircuit()
    error = 1 - np.absolute(np.vdot(wanted, got))**2
    individual.setCMW(error)
    cost = individual.evaluateCost()
    if verbose:
        print("Wanted state is:", wanted)
        print("Produced state is", got)
        print("Error is:", error)
    return (error,cost)

def evaluateInd(individual, verbose=False):
    """
    This function should take an individual,possibly an instance of Candidate class,
    and return a tuple where each element of the tuple is an objective.
    An example objective would be (error,circuitLen) where:
      error = |1 - < createdState | wantedState >
      circuitLen = len(individual.circuit) / MAX_CIRCUIT_LENGTH
      MAX_CIRCUIT_LENGTH is the expected circuit length for the problem.
    """
    wanted = desiredState()
    got = individual.simulateCircuit()
    error = 1 - np.absolute(np.vdot(wanted, got))**2
    individual.setCMW(error)
    if verbose:
        print("Wanted state is:", wanted)
        print("Produced state is", got)
        print("Error is:", error)
    if len(individual.circuit) > 0 and len(individual.circuit) < MAX_CIRCUIT_LENGTH:
        return (error, len(individual.circuit) / MAX_CIRCUIT_LENGTH)
    else:
        return (error, 1.0)


# Your main function
if __name__ == "__main__":
    """
    You should initialize:
      numberOfQubits : number of qubits to be used for the problem
      allowedGates   : allowed set of gates. Default is [Rz,SX,X,CX]
      problemName    : output of the problem will be stored at ./outputs/problemName.txt
      problemDescription : A small header describing the problem.
      fitnessWeights : A tuple describing the weight of each objective. A negative
        weight means that objective should be minimized, a positive weight means
        that objective should be maximized. For example, if you want to represent
        your weights as (error,circuitLen) and want to minimize both with equal
        weight you can just define fitnessWeights = (-1.0,-1.0). Only the relative
        values of the weights have meaning. BEWARE that they are multiplied and summed
        up while calculating the total fitness, so you might want to normalize them.
    """
    # Initialize your variables
    numberOfQubits = 5
    # Let's try to use the basis gate of IBM Quantum Computers
    allowedGates = [X, SqrtX, CNOT, Rz, Swap]
    problemName = "kindaarbitrary"
    problemDescription = "Kind of Arbitrary State initalization for:\n"
    #problemDescription += str(c0) + "|00>" + str(c1) + "|01>" + str(c2) + "|10>" + str(c3) + "|11>\n"
    problemDescription += "numberOfQubits=" + str(numberOfQubits) + "\n"
    problemDescription += "allowedGates=" + str(allowedGates) + "\n"
    # trying to minimize error and length !
    fitnessWeights = (-1.0, -1.0)

    # Create the type of the individual
    creator.create("FitnessMin", base.Fitness, weights=fitnessWeights)
    creator.create("Individual", Candidate, fitness=creator.FitnessMin)
    # Initialize your toolbox and population
    toolbox = base.Toolbox()
    toolbox.register("individual", creator.Individual, numberOfQubits, allowedGates)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Register the necessary functions to the toolbox
    toolbox.register("mate", crossoverInd, toolbox=toolbox)
    toolbox.register("mutate", mutateInd)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("selectAndEvolve", selectAndEvolve)
    toolbox.register("evaluate", evaluateInd)

    # Get it running
    NGEN = 100  # For how many generations should the algorithm run ?
    POPSIZE = 50  # How many individuals should be in the population ?
    verbose = False  # Do you want functions to print out information.
    # Note that they will print out a lot of things.

    # Initialize a random population
    pop = toolbox.population(n=POPSIZE)
    # Run the genetic algorithm

    f = open("./outputs/populations/testpop", "rb")
    pop = pickle.load(f)
    f.close()

    pop = geneticAlgorithm(pop, toolbox, NGEN, problemName, problemDescription, epsilon, verbose=verbose)

    f = open("./outputs/populations/testpop", "wb")
    pickle.dump(pop, f)
    f.close()


