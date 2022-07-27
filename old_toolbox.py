# this file is used when analyzing old pickled python data

from deap import creator, base, tools

from constants import NUMBER_OF_QUBITS, ALLOWED_GATES, CONNECTIVITY, FITNESS_WEIGHTS
from evolution import mutate_ind, select_and_evolve, mate
from tools import evaluate_cost
from old_candidate import Candidate

creator.create("FitnessMin", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("Individual", Candidate, fitness=creator.FitnessMin)


def initialize_toolbox(desired_state):
    toolbox = base.Toolbox()

    toolbox.register("individual", creator.Individual,
                     NUMBER_OF_QUBITS, ALLOWED_GATES, CONNECTIVITY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", mate, toolbox=toolbox)
    toolbox.register("mutate", mutate_ind)
    toolbox.register("select", tools.selSPEA2)
    toolbox.register("selectAndEvolve", select_and_evolve)
    toolbox.register("evaluate", lambda individual: evaluate_cost(
        desired_state, individual))

    return toolbox
