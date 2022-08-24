"""Main program for running the genetic algorithm for a particular state
"""

import os
import argparse
import time
from constants import NUMBER_OF_GENERATIONS, NUMBER_OF_QUBITS, POPULATION_SIZE, ALLOWED_GATES, SAVE_RESULT, BASIS_GATES
from evolution import genetic_algorithm
from tools import load_state, lrsp_circs, save
from toolbox import initialize_toolbox  # also initializes creator

directory = f"performance_data/{NUMBER_OF_QUBITS}QB/{POPULATION_SIZE}POP/"


def main():
    """Runs the genetic algorithm based on the global constants
    """

    # Initialize parser
    parser = argparse.ArgumentParser()

    # Adding optional argument
    parser.add_argument("-p", "--POPSIZE", help="Size of the population")
    parser.add_argument("-g", "--NGEN", help="The number of generations")
    parser.add_argument("-q", "--NQUBIT", help="The number of qubits")
    parser.add_argument("-i", "--INDEX", help="Index of desired state")
    # FIXME -id is illegal (it means -i -d)
    parser.add_argument("-id", "--ID", help="ID of the saved file")

    # Read arguments from command line
    args = parser.parse_args()

    population_size = int(args.POPSIZE) if args.POPSIZE else POPULATION_SIZE
    number_of_generations = int(
        args.NGEN) if args.NGEN else NUMBER_OF_GENERATIONS
    number_of_qubits = int(args.NQUBIT) if args.NQUBIT else NUMBER_OF_QUBITS
    state_index = int(args.INDEX)
    save_file_id = int(args.ID) if args.ID else int(
        len(os.listdir(directory)) / 2)

    state_name = str(number_of_qubits)+"QB_state"+str(state_index)
    problem_name = f"{save_file_id}-{number_of_generations}GEN-{state_name}"

    problem_description = f"""State initalization for:
numberOfQubits={number_of_qubits}
allowedGates={ALLOWED_GATES}"""

    desired_state = load_state(number_of_qubits, state_index).data
    toolbox = initialize_toolbox(desired_state)
    pop = toolbox.population(n=population_size)
    unaltered = lrsp_circs(desired_state, toolbox, FAKE_MACHINE)
    for i, ind in enumerate(unaltered):
        pop[i] = ind

    start = time.perf_counter()
    pop, logbook = genetic_algorithm(pop, toolbox, number_of_generations,
                                     problem_name, problem_description)
    runtime = round(time.perf_counter() - start, 2)

    if SAVE_RESULT:
        # Save the results
        save(pop, logbook, directory, problem_name)
        print(
            f"The population and logbook were saved in {directory}{problem_name}")

    print(f'Runtime: {runtime}s')


if __name__ == '__main__':
    main()
