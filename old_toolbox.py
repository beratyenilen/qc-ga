# this file is used when analyzing old pickled python data

from deap import creator, base, tools
import numpy as np

from constants import NUMBER_OF_QUBITS, ALLOWED_GATES, CONNECTIVITY, FITNESS_WEIGHTS
from evolution import mutate_ind, select_and_evolve, mate
from old_candidate import Candidate

creator.create("FitnessMin", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", Candidate, fitness=creator.FitnessMin)


def evaluate_cost_old(desired_state, candidate):
    """This returns a tuple where each element is an objective. An example
    objective would be (error,circuitLen) where: error = |1 - < createdState |
    wantedState > circuitLen = len(candidate.circuit) / MAX_CIRCUIT_LENGTH
    MAX_CIRCUIT_LENGTH is the expected circuit length for the problem.
    """
    got = candidate.simulateCircuit()
    error = 1 - np.absolute(np.vdot(desired_state, got))**2
    candidate.setCMW(error)
    cost = candidate.evaluateCost()
    return (error, cost)


def initialize_toolbox(desired_state):
    toolbox = base.Toolbox()

    toolbox.register("individual", creator.Individual,
                     NUMBER_OF_QUBITS, ALLOWED_GATES, CONNECTIVITY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", mate, toolbox=toolbox)
    toolbox.register("mutate", mutate_ind)
    toolbox.register("select", tools.selSPEA2)
    toolbox.register("selectAndEvolve", select_and_evolve)
    toolbox.register("evaluate", lambda candidate: evaluate_cost_old(
        desired_state, candidate))

    return toolbox
