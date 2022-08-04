"""This script loads the analysis data and creates the plots in `plots/5QB-final-plots/`
"""
from numpy import *

from qiskit import *
from deap import *
import pandas as pd
from matplotlib import pyplot as plt

from constants import NOISE_MODEL, FAKE_MACHINE
from tools import *
from plot_tools import plot_cnots_noise_fid_scatter, plot_fid_cnot_count, plot_fid_avg_ent, plot_improvement_avg_ent, avg_ent, avg_2qb_ent, plot_cnots_fid_scatter, theoretical_model
from analysis_tools import create_datadir, get_latest_datadir, load_files_by_name, load_states, load_states_with_pop


def main():
    """Plots and saves genetic algorithm results. Creates the plots in `plots/5QB-final-plot_N/`:
    - fid from LRSP and GA (mean entanglement)
    - fid from LRSP and GA (average entropy of 2qb subspaces)
    - fid from LRSP and GA (CNOT count)
    - absolute improvement in fid from LRSP to GA
    - TODO
    """

    ga_datadir = get_latest_datadir("5QB-GA-nondominated-noisy-data")
    print(f"loading GA data from {ga_datadir}")
    ga_nondominated_noisy_fids_purities = load_files_by_name(ga_datadir)
    lrsp_datadir = get_latest_datadir("5QB-LRSP-noisy-fids")
    print(f"loading LRSP data from {lrsp_datadir}")
    lrsp_noisy_fids = load_files_by_name(lrsp_datadir)

    states = {name: state
              for name, state in load_states().items()
              if name != "5QB_state41"}
    print(f"loading population data from performance_data/5QB/400POP/500-30000GEN-*")
    states_with_pop = load_states_with_pop(
        "performance_data/5QB/400POP", "500-30000GEN", states)

    # the analysis dictionary contains fidelity, cnot and purity information of
    # all maximum fidelity individuals per state in GA and LRSP circuits
    max_fid_analysis = {}

    for name, results in list(ga_nondominated_noisy_fids_purities.items()):
        ga_max_fids = []
        for obj in results:
            cnots = obj['cnots']
            runs = obj['runs']
            ind_df = pd.DataFrame(runs)
            max_fid_ind = ind_df[ind_df.noisy_fid == ind_df.noisy_fid.max()]
            ga_max_fids.append({
                'cnots': cnots,
                'noisy_fids': max_fid_ind.noisy_fid.max(),
                'purity': max_fid_ind.purity.max().real
            })
        df = pd.DataFrame(ga_max_fids)

        ga_max_fids = df.groupby('cnots').apply(max)

        data_lrsp = []
        for obj in lrsp_noisy_fids[name]:
            runs = obj['runs']
            ind_df = pd.DataFrame(runs)
            max_fid_ind = ind_df[ind_df.noisy_fid == ind_df.noisy_fid.max()]

            data_lrsp.append({
                'cnots': total_cnots(obj['ind'].circuit),
                'noisy_fids': max_fid_ind.noisy_fid.max(),
                'purity': max_fid_ind.purity.max().real
            })
        df = pd.DataFrame(data_lrsp)
        lrsp_max_fids = df.groupby('cnots').apply(max)

        max_fid_analysis[name] = {'ga_max_fids': ga_max_fids,
                                  'lrsp_max_fids': lrsp_max_fids}

    states_with_avg_ent = {}
    states_with_avg_2qb_ent = {}
    for name, state in states.items():
        states_with_avg_ent[name] = avg_ent(state)
        states_with_avg_2qb_ent[name] = avg_2qb_ent(state)

    plot_dir = create_datadir("plots/5QB-final-plots")

    title = 'fid from LRSP and GA (CNOT count)'
    plot_fid_cnot_count(states, max_fid_analysis, title)
    plt.savefig(path.join(plot_dir, title))

    title = 'fid from LRSP and GA (mean entanglement)'
    plot_fid_avg_ent(states_with_avg_ent, max_fid_analysis, title,
                     xlabel='Mean entanglement of a single qubit')
    plt.savefig(path.join(plot_dir, title))

    title = 'fid from LRSP and GA (average entropy of 2qb subspaces)'

    plot_fid_avg_ent(states_with_avg_2qb_ent, max_fid_analysis, title,
                     xlabel='Mean entanglement of two qubits')
    plt.savefig(path.join(plot_dir, title))

    title = 'absolute improvement in fid from LRSP to GA'
    plot_improvement_avg_ent(states_with_avg_ent, max_fid_analysis, title)
    plt.savefig(path.join(plot_dir, title))

    title = 'fid from GA'
    for state, obj in states_with_pop.items():
        pop = obj['pop']
        plot_cnots_fid_scatter(pop)
    theoretical_model(p=0)
    plt.savefig(path.join(plot_dir, title))

    title = 'noisy fid from GA'
    for state, obj in states_with_pop.items():
        pop = obj['pop']
        plot_cnots_noise_fid_scatter(ga_nondominated_noisy_fids_purities)
    theoretical_model(p=0)
    plt.savefig(path.join(plot_dir, title))

    print(f"Plots saved in {plot_dir}")


if __name__ == '__main__':
    main()
