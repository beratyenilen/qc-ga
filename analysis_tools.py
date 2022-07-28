
import os
import pickle
import re
from threading import Thread

from qiskit import execute

from constants import CONNECTIVITY, NOISE_MODEL


def load_files_by_name(basedir):
    loaded = {}
    for name in next(os.walk(basedir))[2]:
        with open(os.path.join(basedir, name), 'rb') as file:
            loaded[name] = pickle.load(file)
    return loaded


def load_states():
    return load_files_by_name('./states/5_qubits')


def load_states_with_pop(basedir, prefix, states):
    """Loads GA populations from `{basedir}/{prefix}-{state name}.pop` by
    iterating over each state in `states`.
    """
    states_with_pop = {}
    for name, state in states.items():
        try:
            with open(os.path.join(basedir, f'{prefix}-{name}.pop'), 'rb') as f:
                states_with_pop[name] = {'state': state, 'pop': pickle.load(f)}
        except FileNotFoundError as e:
            print(f"warning: {e}")
    return states_with_pop


def uniq_by(iterator, uniq_fn):
    """Returns a list of all elements from `iterator` which after being
    mapped by `uniq_fn` are unique.
    """
    uniq_so_far = []
    uniq_so_far_mapped = []
    for elem in iterator:
        mapped = uniq_fn(elem)
        if mapped not in uniq_so_far_mapped:
            uniq_so_far_mapped.append(mapped)
            uniq_so_far.append(elem)
    return uniq_so_far


def increment_suffix(name):
    """Returns the filename with the suffix incremented. The suffix is a number
    after an underscore: `{basename}_{N}`. If `name` has no suffix, N defaults
    to 0.
    """
    basename, index = re.match(
        r'^(5QB[A-Za-z-]+)_?(\d+?)?$', name).groups()
    return f'{basename}_{int(index or 0) + 1}'


def create_datadir(datadir):
    """Creates a new directory with incrementing suffix, returning the name of
    the new directory"""
    while os.path.exists(datadir):
        datadir = increment_suffix(datadir)

    os.mkdir(datadir)
    return datadir


def multithread_chunks(data, chunks, run):
    treds = []
    for i in range(chunks):
        chunk = list(data)[i::chunks]
        print(len(chunk))
        t = Thread(target=run, args=[chunk])
        t.start()
        treds.append(t)

    for t in treds:
        t.join()


def noisy_simulation_density_matrix(backend, circ):
    circ.save_density_matrix(label='density_matrix')
    result = execute(circ, backend,
                     coupling_map=CONNECTIVITY,
                     noise_model=NOISE_MODEL).result()
    return result.data()['density_matrix']
