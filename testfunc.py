from candidate import *
import numpy as np


def getPermutationMatrix(self):
  '''
  Args:
      perm: a list representing where each qubit is mapped to. 
          perm[i] represents the logical qubit and i represents the physical qubit.
          So [1,0,2] will be interpreted as (physical<-logical) 0<-1, 1<-0, 2<-2
  Returns:
      2^N x 2^N numpy matrix representing the action of the permutation where
      N is the number of qubits.
  '''
  Nexp = 2**self.numberOfQubits
  M = np.zeros((Nexp,Nexp))
  toBin = list(bin(0)[2:].zfill(self.numberOfQubits))
  for frm in range(Nexp):
    frmBin = list(bin(frm)[2:].zfill(self.numberOfQubits))
    for p in range(self.numberOfQubits):
      toBin[p] = frmBin[self.perm[p]]
      to = int(''.join(toBin),2)
    M[to][frm] = 1.0
  return M

def getPermutation(perm,numberOfQubits):
  """
  Returns the permutation as a list of tuples. [(logical,physical),(log,phys),...]
  """
  res = []
  res.append(("log","phy"))
  for i in range(numberOfQubits):
    res.append((perm[i],i))
  return res