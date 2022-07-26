from deap import creator, base, tools

from constants import NUMBER_OF_QUBITS, ALLOWED_GATES, CONNECTIVITY, FITNESS_WEIGHTS
from evolution import mutate_ind, select_and_evolve, mate
from tools import evaluate_cost
from individual import Individual

# the following configuration is top-level due to it being required whenever the
# toolbox is used
creator.create("fitness_min", base.Fitness, weights=FITNESS_WEIGHTS)
creator.create("individual", Individual, fitness=creator.fitness_min)

def initialize_toolbox(desired_state):
    """Initializes the DEAP methods, like mate, mutate, select_and_evolve
    """
    toolbox = base.Toolbox()

    toolbox.register("individual", creator.individual,
                     NUMBER_OF_QUBITS, ALLOWED_GATES, CONNECTIVITY)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", mate, toolbox=toolbox)
    toolbox.register("mutate", mutate_ind)
    toolbox.register("select", tools.selNSGA2)
    toolbox.register("select_and_evolve", select_and_evolve)
    toolbox.register("evaluate", lambda individual: evaluate_cost(desired_state, individual))

    return toolbox
