from candidate import *
import numpy as np


def getPermutationMatrix(perm, numberOfQubits):
    """
    Args:
        perm: a list representing where each qubit is mapped to.
            perm[i] represents the physical qubit and i represents the virtual qubit.
            So [1,0,2] will be interpreted as (virtual->physical) 0->1, 1->0, 2->2
    Returns:
        2^N x 2^N numpy matrix representing the action of the permutation where
        N is the number of qubits.
    """
    Nexp = 2 ** numberOfQubits
    M = np.zeros((Nexp, Nexp))
    for frm in range(Nexp):
        toBin = list(bin(0)[2:].zfill(numberOfQubits))
        frmBin = list(bin(frm)[2:].zfill(numberOfQubits))
        toBin.reverse()
        frmBin.reverse()
        for p in range(numberOfQubits):
            toBin[p] = frmBin[perm.index(p)]
        toBin.reverse()
        to = int("".join(toBin), 2)
        M[to][frm] = 1.0
    return M


def getPermutation(perm, numberOfQubits):
    """
    Returns the permutation as a list of tuples. [(logical,physical),(log,phys),...]
    """
    res = []
    res.append(("log", "phy"))
    for i in range(numberOfQubits):
        res.append((perm[i], i))
    return res


perm = [0, 1, 2]
n = 3
m = getPermutationMatrix(perm, n)
