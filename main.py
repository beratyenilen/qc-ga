import os
import argparse
from datetime import datetime
import time
from deap import creator, base
from individual import Individual
from constants import NUMBER_OF_GENERATIONS, NUMBER_OF_QUBITS, POPULATION_SIZE, VERBOSE, ALLOWED_GATES, SAVE_RESULT
from evolution import genetic_algorithm
from tools import initialize_toolbox, load_state, lrsp_circs, save


directory = f"performance_data/{NUMBER_OF_QUBITS}QB/{POPULATION_SIZE}POP/"
ID = int(len(os.listdir(directory)) / 2)

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
    POPULATION_SIZE = int(args.POPSIZE)
if args.NGEN:
    NUMBER_OF_GENERATIONS = int(args.NGEN)
if args.NQUBIT:
    NUMBER_OF_QUBITS = int(args.NQUBIT)
if args.INDEX:
    STATE_INDEX = int(args.INDEX)
if args.ID:
    ID = int(args.ID)
    load_file = True

state_name = str(NUMBER_OF_QUBITS)+"QB_state"+str(STATE_INDEX)
problem_name = f"{ID}-{NUMBER_OF_GENERATIONS}GEN-{state_name}"

now = datetime.now()
time_str = now.strftime("%d.%m.%y-%H:%M")

problem_description = "State initalization for:\n"
problem_description += "numberOfQubits=" + str(NUMBER_OF_QUBITS) + "\n"
problem_description += "allowedGates=" + str(ALLOWED_GATES) + "\n"

# trying to minimize error and length !
fitness_weights = (-1.0, -0.5)

# Create the type of the individual
# TODO move to utils
creator.create("fitness_min", base.Fitness, weights=fitness_weights)
creator.create("individual", Individual, fitness=creator.fitness_min)


def main():
    # Initialize your toolbox and population
    desired_state = load_state(NUMBER_OF_QUBITS, STATE_INDEX).data
    toolbox = initialize_toolbox(desired_state)
    pop = toolbox.population(n=POPULATION_SIZE)
    unaltered = lrsp_circs(desired_state, toolbox)
    for i in range(len(unaltered)):
        pop[i] = unaltered[i]

    start = time.perf_counter()
    pop, logbook = genetic_algorithm(pop, toolbox, NUMBER_OF_GENERATIONS, problem_name,
                                     problem_description, verbose=VERBOSE)
    runtime = round(time.perf_counter() - start, 2)

    # Save the results
    if SAVE_RESULT:
        save(pop, logbook, directory, problem_name)
        print(
            f"The population and logbook were saved in {directory}{problem_name}")

    print(f'Runtime: {runtime}s')
    return runtime


if __name__ == '__main__':
    main()
