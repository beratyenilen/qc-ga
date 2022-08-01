from matplotlib import pyplot as plt
import numpy as np
from deap.tools.emo import sortNondominated
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Operator, state_fidelity
from qiskit.circuit.library import Permutation


def plot_fid_cnot_count(states, analysis, title):
    plt.figure(figsize=(10, 7))
    plt.title(title)

    for name, _ in states.items():
        data = analysis[name]
        ga_max_fids = data['ga_max_fids']
        lrsp_max_fids = data['lrsp_max_fids']

        ga_best_cnot = ga_max_fids.cnots[ga_max_fids.noisy_fids.eq(
            ga_max_fids.noisy_fids.max())].max()
        lrsp_best_cnot = lrsp_max_fids.cnots[lrsp_max_fids.noisy_fids.eq(
            lrsp_max_fids.noisy_fids.max())].max()
        plt.scatter(x=ga_best_cnot,
                    y=ga_max_fids.noisy_fids.max(), color='red')
        plt.scatter(x=lrsp_best_cnot,
                    y=lrsp_max_fids.noisy_fids.max(), color='blue', marker='.')
    plt.ylim(0, 1)
    plt.ylabel('Fidelity')
    plt.xlabel('CNOT count')


def plot_fid_avg_ent(states_with_avg_ent, analysis, title, xlabel):
    plt.figure(figsize=(10, 7))
    plt.title(title)
    for name, avg_ent in states_with_avg_ent.items():
        data = analysis[name]
        ga_max_fids = data['ga_max_fids']
        lrsp_max_fids = data['lrsp_max_fids']
        plt.scatter(x=avg_ent, y=ga_max_fids.noisy_fids.max(), color='red')
        plt.scatter(x=avg_ent, y=lrsp_max_fids.noisy_fids.max(),
                    color='blue', marker='.')
    plt.ylim(0, 1)
    plt.ylabel('Fidelity')
    plt.xlabel(xlabel)


def plot_improvement_avg_ent(states_with_avg_ent, analysis, title):
    plt.figure(figsize=(10, 7))
    plt.title(title)
    for name, avg_ent in states_with_avg_ent.items():
        data = analysis[name]
        ga_max_fids = data['ga_max_fids']
        lrsp_max_fids = data['lrsp_max_fids']

        plt.scatter(x=avg_ent, y=data['ga_max_fids'].noisy_fids.max(
        ) - data['lrsp_max_fids'].noisy_fids.max(), color='green', marker='.')

    plt.ylabel('Fidelity difference')
    plt.xlabel('Mean entanglement of a single qubit')

# ====================== UNUSED ======================


def plotFitSize(logbook, fitness="min", size="avg"):
    """Plot the fitness and size

    Values for fitness and size:
        "min" plots the minimum
        "max" plots the maximum
        "avg" plots the average
        "std" plots the standard deviation
    """
    gen = logbook.select("gen")
    fit_mins = logbook.chapters["fitness"].select(fitness)
    size_avgs = logbook.chapters["size"].select(size)

    _fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, fit_mins, "b-", label=f"{fitness} Error")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Error", color="b")
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    plt.ylim(0, 1)

    ax2 = ax1.twinx()
    line2 = ax2.plot(gen, size_avgs, "r-", label=f"{size} Size")
    ax2.set_ylabel("Size", color="r")
    for tl in ax2.get_yticklabels():
        tl.set_color("r")

    lns = line1 + line2
    labs = [line.get_label() for line in lns]
    ax1.legend(lns, labs, loc="center right")


def plotCircLengths(circs, circs2):
    sizes1 = np.array([circ.size() for circ in circs])
    max_size = sizes1.max()
    sizes2 = np.array([circ.size() for circ in circs2])
    if sizes2.max() > max_size:
        max_size = sizes2.max()
    plt.hist(sizes1, bins=max_size, range=(0, max_size), alpha=0.5)


def plotLenFidScatter(pop):
    data = []
    for circ in pop:
        data.append([circ.toQiskitCircuit().size(),
                    1 - circ.fitness.values[0]])

    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, marker='.')
    plt.ylabel("Fidelity")
    plt.xlabel("Length")
    plt.ylim(0, 1)


def fitfidScatter(pop, color='red', plot_all=True):
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    data = []
    for i in range(len(ranks[0]), len(pop)):
        pop[i].trim()
        circ = pop[i]
        data.append([circ.fitness.values[1], 1 - circ.fitness.values[0]])
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    if (plot_all):
        plt.scatter(x, y, color='b', marker='.')
    data = []
    data = np.array(
        [[circ.fitness.values[1], 1 - circ.fitness.values[0]] for circ in front])
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color=color, marker='.')
    plt.ylabel("Fidelity")
    plt.xlabel("Cost")
    plt.ylim(0, 1)


def fitNoisefidScatter(pop, state_vector, fake_machine, noise_model, plot_all=True):
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    x = []
    y = []
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]

    if (plot_all):
        front = pop

    for circ in front:
        # TODO refactor
        x.append(circ.fitness.values[1])
        permutation = circ.getPermutationMatrix()
        permutation = np.linalg.inv(permutation)
        circ = circ.toQiskitCircuit()
        circ.measure_all()
        circ = transpile(circ, fake_machine, optimization_level=0)
        permutation2 = getPermutation(circ)
        # Creating a circuit for qubit mapping
        perm_circ = Permutation(5, permutation2)
        perm_unitary = Operator(perm_circ)  # Matrix for the previous circuit

        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(
            result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(perm_unitary @ permutation @
                 state_vector, density_matrix_noisy))

    plt.ylabel("Fidelity")
    plt.xlabel("Cost")
    plt.ylim(0, 1)
    return x, y


def plotCNOTSFidScatter(pop):
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    data = []
    for ind in front:
        # TODO refactor with totalCNOTs
        circ = ind.circuit
        l = len(circ)
        if (l == 0):
            continue
        cnots = 0
        for gate in circ:
            if (gate[1] == CNOT):
                cnots += 1
        data.append([cnots, 1 - ind.fitness.values[0]])
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.ylabel("Fidelity")
    plt.xlabel("CNOTS")
    plt.ylim(0, 1)
    return x, y


def plotCNOTSNoiseFidScatter(pop, state_vector, fake_machine, noise_model, color='red'):
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)

    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    x = []
    y = []
    for ind in front:
        circ = ind.circuit
        # TODO refactor with totalCNOTs
        if len(circ) == 0:
            continue
        cnots = 0
        for gate in circ:
            if gate[1] == CNOT:
                cnots += 1
        permutation = ind.getPermutationMatrix()
        permutation = np.linalg.inv(permutation)
        circ = ind.toQiskitCircuit()
        circ.measure_all()
        circ = transpile(circ, fake_machine, optimization_level=0)
        permutation2 = getPermutation(circ)
        # Creating a circuit for qubit mapping
        perm_circ = Permutation(5, permutation2)
        perm_unitary = Operator(perm_circ)  # Matrix for the previous circuit

        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(
            result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(perm_unitary @ permutation @
                 state_vector, density_matrix_noisy))
        x.append(cnots)
    plt.ylabel("Fidelity")
    plt.xlabel("CNOTS")
    plt.ylim(0, 1)
    plt.scatter(x, y, color=color, marker='.')


def plotLenCNOTScatter(pop, color='red'):
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    data = []

    # TODO refactor with totalCNOTs
    for ind in front:
        circ = ind.circuit
        l = len(circ)
        if l == 0:
            continue
        cnots = 0
        for gate in circ:
            if gate[1] == CNOT:
                cnots += 1
        data.append([l, cnots])
    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color=color, marker='.')
    plt.ylabel("CNOTS")
    plt.xlabel("Length")


def paretoFront(pop, color='red', plot_all=True):
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    data = []
    for i in range(len(ranks[0]), len(pop)):
        pop[i].trim()
        circ = pop[i]
        data.append([circ.toQiskitCircuit().size(),
                    1 - circ.fitness.values[0]])

    data = np.array(data)
    x = data[:, 0]
    y = data[:, 1]
    if plot_all:
        plt.scatter(x, y, color='b', marker='.')
    data = []
    data = np.array([[circ.toQiskitCircuit().size(), 1 -
                    circ.fitness.values[0]] for circ in front])

    x = data[:, 0]
    y = data[:, 1]
    plt.scatter(x, y, color=color, marker='.')
    plt.ylabel("Fidelity")
    plt.xlabel("Length")
    plt.ylim(0, 1)


def theoreticalModel(n=5, p=0.01, l_lim=35, L=10, Nph=2.5, label=''):
    # p: Probability of error
    # L: Pairs of qubits connected by CNOT gates
    d = 2**n-n-1

    c1 = -(2*np.log(2)*n+np.log(L))/d
    c2 = -np.log(2)*n**2/d

    l = np.linspace(0, l_lim, 1000)

    y = ((1-p)**l)*(1-np.exp(c1*(l/Nph)+c2))
    plt.plot(l, y, label=label)


def paretoNoiseFids(pop, state_vector, fake_machine, noise_model, plot_all=True, color='red'):
    backend = AerSimulator(method='density_matrix', noise_model=noise_model)
    # backend = AerSimulator(method='density_matrix')

    x = []
    y = []
    ranks = sortNondominated(pop, len(pop), first_front_only=True)
    front = ranks[0]
    if (plot_all):
        front = pop

    for circ in front:
        permutation = circ.getPermutationMatrix()
        permutation = np.linalg.inv(permutation)
        circ = circ.toQiskitCircuit()
        circ.measure_all()
        circ = transpile(circ, fake_machine, optimization_level=0)
        permutation2 = getPermutation(circ)
        # Creating a circuit for qubit mapping
        perm_circ = Permutation(5, permutation2)
        perm_unitary = Operator(perm_circ)  # Matrix for the previous circuit

        circ.snapshot_density_matrix('density_matrix')
        result = backend.run(circ).result()
        density_matrix_noisy = DensityMatrix(
            result.data()['snapshots']['density_matrix']['density_matrix'][0]['value'])
        y.append(state_fidelity(perm_unitary @ permutation @
                 state_vector, density_matrix_noisy))
        x.append(circ.size())
    plt.ylabel("Fidelity")
    plt.xlabel("Length")
    plt.ylim(0, 1)
    plt.scatter(x, y, color=color, marker='.')
