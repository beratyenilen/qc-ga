"""This script loads the analysis data and creates the plots in TODO
"""

from threading import Thread
from old_candidate import Candidate
from old_toolbox import initialize_toolbox
from tools import *
import qiskit.quantum_info as qi
from qiskit.quantum_info import Operator
from qiskit.circuit.library import Permutation
from qiskit import *
from deap.tools.emo import sortNondominated
from deap import *
from qiskit.quantum_info import state_fidelity, purity
import pandas as pd
from matplotlib import pyplot as plt
from numpy import *
import pickle


# import old toolbox and individual for analyzing old data


def load_files_by_name(basedir):
    loaded = {}
    for name in next(os.walk(basedir))[2]:
        with open(os.path.join(basedir, name), 'rb') as file:
            loaded[name] = pickle.load(file)
    return loaded


basedir = 'states/5_qubits'


def run_init_lrsp_circs(data):
    for name, state in data:
        circs = LRSP_circs(state, initialize_toolbox(state))
        ranks = sortNondominated(circs, len(circs), first_front_only=True)
        front = ranks[0]
        # states_with_pop['lrsp_front'] = front
        f = open('5QB-LRSP-fronts/'+name, 'wb')
        pickle.dump(front, f)
        f.close()
        print(f'finished {name}')


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


def uniqBy(l, f):
    uniq_so_far = []
    uniq_so_far_mapped = []
    for c in l:
        d = f(c)
        if not d in uniq_so_far_mapped:
            uniq_so_far_mapped.append(d)
            uniq_so_far.append(c)
    return uniq_so_far


if __name__ == '__main__':
    # ======================= load pickled files into memory ====================

    # TODO plot code to plot_tools
