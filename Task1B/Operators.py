"Operators"

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
from dynamiqs import destroy, expm, QArray

def displacement(N, alpha):
    '''
    N (int) = hilbert space dimension
    alpha (complex number) = displacement

    returns (QArray) displacement operator with value alpha
    '''
    a = destroy(N)
    generator = alpha * a.dag() - jnp.conj(alpha) * a

    # exponential
    D = expm(generator) 
    return D

def parity(N):
    '''
    N (int) = hilbert space dimension

    returns (QArray) parity operator
    '''
    a = destroy(N)
    i = 1j
    generator = i * jnp.pi * a @ a.dag()
    P = expm(generator)
    return P

# def observable(N, alpha):
#     '''
#     N (int) = hilbert space dimension
#     alpha (complex number) = displacement

#     returns (QArray) the observable as defined in the challenge file
#     '''
#     D = displacement(N, alpha)
#     P = parity(N)
#     E = 1/2 * (np.eye(N) + D @ P @ D.conj().mT)
#     return E


def observable(N, alpha):

    #To Increase the Hilbert Space size before truncation
    if 2*N < 10:    
        N_big = N + 10
    else:
        N_big = 2*N            
                                              
    D_big = displacement(N_big, alpha)
    P_big = parity(N_big)
    E_big = 0.5 * (jnp.eye(N_big) + D_big @ P_big @ D_big.conj().mT)


    # Truncate back
    E = E_big.data[:N, :N]

    return E
