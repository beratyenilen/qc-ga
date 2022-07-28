# TODO combine with ga-nondom...analysis

import pickle
from os import path

import numpy as np
from qiskit import transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.circuit.library import Permutation
from qiskit.quantum_info import Operator
from qiskit.quantum_info import state_fidelity, purity
from deap.tools.emo import sortNondominated

from analysis_tools import create_datadir, load_states, load_files_by_name, multithread_chunks, noisy_simulation_density_matrix, uniq_by, load_states_with_pop
from constants import FAKE_MACHINE, NOISE_MODEL, BASIS_GATES
from tools import get_permutation, get_permutation_new_and_improved, total_cnots, lrsp_circs
from old_toolbox import initialize_toolbox

NUMBER_OF_SIMULATIONS = 1  # 100 is a sensible default

# How many threads to spawn, chunks to split the simulations into
NUMBER_OF_TASKS = 4

# Set to None to run simulations for all individuals
LIMIT_INDIVIDUALS = 1

if __name__ == '__main__':
    states = load_states()
    backend = QasmSimulator(method='density_matrix', noise_model=NOISE_MODEL)
    permutation_matrix_cache = {}

    def run_init_lrsp_fronts(datadir, chunk):
        for name, state in chunk:
            circs = lrsp_circs(state, initialize_toolbox(state), BASIS_GATES)
            ranks = sortNondominated(circs, len(circs), first_front_only=True)
            front = ranks[0]
            with open(path.join(datadir, name), 'wb') as file:
                pickle.dump(front, file)
            print(f'run_init_lrsp_fronts: finished {name}')

    # TODO don't init lrsp fronts if they already exist
    lrsp_fronts_datadir = create_datadir('5QB-LRSP-fronts')
    multithread_chunks(states.items(),
                       NUMBER_OF_TASKS,
                       lambda chunk: run_init_lrsp_fronts(lrsp_fronts_datadir, chunk))

    lrsp_fronts = load_files_by_name('5QB-LRSP-fronts')

    lrsp_fronts_uniq = {name: uniq_by(circs, lambda c: c.circuit)
                        for name, circs in lrsp_fronts.items()}

    assert states.keys() == lrsp_fronts_uniq.keys()

    # ====================== 5QB-LRSP-noisy-fids ========================

    def run_lrsp_noisy_fids(datadir, chunk):
        for name, state in chunk:
            circs = lrsp_fronts_uniq[name]

            data = []
            for ind in circs[0:LIMIT_INDIVIDUALS]:
                permutation = ind.getPermutationMatrix()
                permutation = np.linalg.inv(permutation)
                cnots = total_cnots(ind.circuit)
                ind_data = {
                    'ind': ind,
                    'cnots': cnots,
                    'runs': []
                }
                for _ in range(NUMBER_OF_SIMULATIONS):
                    circ = ind.toQiskitCircuit()
                    circ.measure_all()
                    circ = transpile(circ, FAKE_MACHINE, optimization_level=0)
                    transpile_permutation = tuple(get_permutation(circ))

                    cached_permutation_matrix = permutation_matrix_cache.get(
                        transpile_permutation)
                    if cached_permutation_matrix is None:
                        # Creating a circuit for qubit mapping
                        perm_circ = Permutation(5, transpile_permutation)
                        cached_permutation_matrix = Operator(
                            perm_circ)  # Matrix for the previous circuit
                        permutation_matrix_cache[transpile_permutation] = cached_permutation_matrix

                    density_matrix_noisy = noisy_simulation_density_matrix(
                        backend, circ)
                    fid = state_fidelity(cached_permutation_matrix @
                                         permutation @
                                         state._data, density_matrix_noisy,
                                         validate=False)
                    ind_data['runs'].append({
                        'noisy_fid': fid,
                        'purity': purity(density_matrix_noisy)
                    })
                data.append(ind_data)

            with open(path.join(datadir, name), 'wb') as file:
                pickle.dump(data, file)
            print(f'run_lrsp_noisy_fids: finished {name}')

    lrsp_noisy_datadir = create_datadir("5QB-LRSP-noisy-fids")
    multithread_chunks(states.items(),
                       NUMBER_OF_TASKS,
                       lambda chunk: run_lrsp_noisy_fids(lrsp_noisy_datadir, chunk))

    # ===================== 5QB-GA-nondominated-noisy-data ====================

    states_with_pop = load_states_with_pop(
        "performance_data/5QB/400POP", "500-30000GEN", states)

    def run_ga_nondominated_noisy_fids(datadir, chunk):
        for name, obj in chunk:
            data = []
            state = obj['state']
            pop = sortNondominated(obj['pop'], len(
                obj['pop']), first_front_only=True)[0]
            pop = uniq_by(pop, lambda c: c.circuit)
            for ind in pop[0:LIMIT_INDIVIDUALS]:
                permutation = ind.getPermutationMatrix()
                permutation = np.linalg.inv(permutation)
                cnots = total_cnots(ind.circuit)
                ind_data = {
                    'ind': ind,
                    'cnots': cnots,
                    'runs': []
                }
                # find the best circuit from noisy results
                for _ in range(NUMBER_OF_SIMULATIONS):
                    circ = ind.toQiskitCircuit()
                    circ.measure_all()
                    circ = transpile(circ, FAKE_MACHINE, optimization_level=0)
                    transpile_permutation = tuple(get_permutation(circ))

                    cached_permutation_matrix = permutation_matrix_cache.get(
                        transpile_permutation)
                    if cached_permutation_matrix is None:
                        # Creating a circuit for qubit mapping
                        perm_circ = Permutation(5, transpile_permutation)
                        cached_permutation_matrix = Operator(
                            perm_circ)  # Matrix for the previous circuit
                        permutation_matrix_cache[transpile_permutation] = cached_permutation_matrix

                    density_matrix_noisy = noisy_simulation_density_matrix(
                        backend, circ)

                    fid = state_fidelity(cached_permutation_matrix @ permutation @
                                         state._data, density_matrix_noisy, validate=False)

                    ind_data['runs'].append(
                        {'noisy_fid': fid, 'purity': purity(density_matrix_noisy)})
                    # ind_data['runs'].append( # FIXME old
                    # {'transpile_permutation': transpile_permutation, 'density_matrix_noisy': density_matrix_noisy})
                data.append(ind_data)

            with open(path.join(datadir, name), 'wb') as file:
                pickle.dump(data, file)
            print(f'run_ga_nondominated_noisy_fids: finished {name}')

    ga_noisy_datadir = create_datadir('5QB-GA-nondominated-noisy-data')
    multithread_chunks(states_with_pop.items(),
                       NUMBER_OF_TASKS,
                       lambda chunk: run_ga_nondominated_noisy_fids(ga_noisy_datadir, chunk))
