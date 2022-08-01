"""This script loads the analysis data and creates the plots in `plots/5QB-final-plots/`
"""
from plot_tools import plot_fid_cnot_count, plot_fid_avg_ent, plot_improvement_avg_ent
from analysis_tools import create_datadir, get_latest_datadir, load_files_by_name, load_states
from tools import *
import qiskit.quantum_info as qi
from qiskit import *
from deap import *
import pandas as pd
from matplotlib import pyplot as plt
from numpy import *
import pickle
import itertools


# import old toolbox and individual for analyzing old data


if __name__ == '__main__':
    # ======================= load pickled files into memory ====================

    # TODO plot code to plot_tools

    """
        This code block plots the maximum fidelities of LRSP and GA with noise for each state individually.
        Variance in the results for a specific individual is due to different permutations that arise from
        qiskit transpile function; the noise from cnots in different connections varies. Also, transpile
        might change/add single qubit gates.
    """

    ga_datadir = get_latest_datadir("5QB-GA-nondominated-noisy-data")
    print(f"loading GA data from {ga_datadir}")
    ga_nondominated_noisy_fids_purities = load_files_by_name(ga_datadir)
    lrsp_datadir = get_latest_datadir("5QB-LRSP-noisy-fids")
    print(f"loading LRSP data from {lrsp_datadir}")
    lrsp_noisy_fids = load_files_by_name(lrsp_datadir)

    analysis = {}

    for name, results in list(ga_nondominated_noisy_fids_purities.items()):
        ga_max_fids = []
        for obj in results:
            cnots = obj['cnots']
            runs = obj['runs']
            ind_df = pd.DataFrame(runs)
            max_fid_ind = ind_df[ind_df.noisy_fid == ind_df.noisy_fid.max()]
            ga_max_fids.append({'cnots': cnots, 'noisy_fids': max_fid_ind.noisy_fid.max(
            ), 'purity': max_fid_ind.purity.max().real})
        df = pd.DataFrame(ga_max_fids)

        ga_max_fids = df.groupby('cnots').apply(max)

        data_lrsp = []
        for obj in lrsp_noisy_fids[name]:
            # ent = first_qubit_entanglement_entropy(states[name])
            runs = obj['runs']
            ind_df = pd.DataFrame(runs)
            max_fid_ind = ind_df[ind_df.noisy_fid == ind_df.noisy_fid.max()]

            data_lrsp.append({'cnots': total_cnots(
                obj['ind'].circuit), 'noisy_fids': max_fid_ind.noisy_fid.max(
            ), 'purity': max_fid_ind.purity.max().real})  # TODO find maximum noisy fid from obj.runs

        df = pd.DataFrame(data_lrsp)
        # df.noisy_fids = df.noisy_fids.apply(max)
        lrsp_max_fids = df.groupby('cnots').apply(max)

        analysis[name] = {'ga_max_fids': ga_max_fids,
                          'lrsp_max_fids': lrsp_max_fids}

    n = 5
    qubits = range(n)

    # TODO plot tools
    def avg_ent(psi):
        return sum([qi.entropy(qi.partial_trace(psi, [q])) / 5 for q in qubits])

    def avg_2qb_ent(psi):
        return sum([qi.entropy(qi.partial_trace(psi, sub)) / 10 for sub in list(itertools.combinations(qubits, 2))])

    def first_qubit_entanglement_entropy(psi):
        rho = qi.partial_trace(psi, [4])
        return qi.entropy(rho)

    states = {name: state for name, state in load_states().items() if name !=
              "5QB_state41"}

    states_with_avg_ent = {}
    states_with_avg_2qb_ent = {}
    for name, state in states.items():
        states_with_avg_ent[name] = avg_ent(state)
        states_with_avg_2qb_ent[name] = avg_2qb_ent(state)

    plot_dir = create_datadir("plots/5QB-final-plots")

    title = 'fid from LRSP and GA (CNOT count)'
    plot_fid_cnot_count(states, analysis, title)
    plt.savefig(path.join(plot_dir, title))

    title = 'fid from LRSP and GA (mean entanglement)'
    plot_fid_avg_ent(states_with_avg_ent, analysis, title,
                     xlabel='Mean entanglement of a single qubit')
    plt.savefig(path.join(plot_dir, title))

    title = 'fid from LRSP and GA (average entropy of 2qb subspaces)'

    plot_fid_avg_ent(states_with_avg_2qb_ent, analysis, title,
                     xlabel='Mean entanglement of two qubits')
    plt.savefig(path.join(plot_dir, title))

    title = 'absolute improvement in fid from LRSP to GA'
    plot_improvement_avg_ent(states_with_avg_ent, analysis, title)
    plt.savefig(path.join(plot_dir, title))
