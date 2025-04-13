"DisCatStateGen"
""""Dissipative Cat State from a Two-Photon Exchange Hamiltonian Generation"""

import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# simulation parameters
n = 5         # Hilbert space size
g= 1
epsilon = -4
kappa = 10
ntsave = 201   # number of saved states
T = 4

# operators
a = dq.tensor(dq.destroy(n), np.eye(n))
b = dq.tensor(np.eye(n), dq.destroy(n))


H = g * a.dag() @ a.dag() @ b + g * a @ a @ b.dag() + epsilon * (b + b.dag())
jump_ops = [jnp.sqrt(kappa) * b]

# initial state
psi0 = dq.tensor(dq.basis(n, 0), (dq.basis(n, 0)))

# save times
tsave = jnp.linspace(0.0, T, ntsave)

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave)
