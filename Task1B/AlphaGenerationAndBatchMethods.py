"AlphaGenerationMethods"

import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

fock = dq.states.fock
state = fock(5, 2)





# xx, pp = np.meshgrid(xvec, yvec, indexing='xy') 


def get_W_at_alphas(xvec, yvec, w, alphas):
    '''
    W (2D array) wigner state values
    xvec (1d array) position axis values
    yvec (1d array) momentum axis values
    alphas (1d array of complex) where to evaluate the wigner function at

    returns (array) values of wigner function at alphas
    '''
    
    x = alphas.real
    y = alphas.imag


    #find closest index to alpha value
    ix = np.abs(xvec[:, None] - x[None, :]).argmin(axis=0)
    iy = np.abs(yvec[:, None] - y[None, :]).argmin(axis=0)

    return w[iy, ix]


# Better sampling. Focus more points towards the center
def nonlinear_linspace(start, stop, num, scale=1.5):
    """Generate nonlinearly spaced values more dense near zero."""
    lin = np.linspace(-1, 1, num)
    stretched = np.sign(lin) * np.abs(lin)**scale
    return np.interp(stretched, [-1, 1], [start, stop])


def alphas_nonlinear_linspace(state):
    xvec, yvec, w = dq.wigner(state)
    # Assume xvec and yvec are already defined from dq.wigner(...)
    xmin, xmax = xvec[0], xvec[-1]
    ymin, ymax = yvec[0], yvec[-1]

    # Generate denser points near zero, constrained within Wigner grid
    n = 30  # number of points along each axis
    x_dense = nonlinear_linspace(xmin, xmax, n, scale=2)
    y_dense = nonlinear_linspace(ymin, ymax, n, scale=2)

    # Create 2D grid of complex alpha values
    xx, yy = np.meshgrid(x_dense, y_dense)
    alphas  =(xx + 1j * yy).flatten() * np.sqrt(2)
    return alphas



def alpha_gen(state):
    xvec, yvec, w = dq.wigner(state)
    dim = len(xvec)
    xx, pp = np.meshgrid(xvec, yvec, indexing='xy') 

    alpha_k = (xx + 1j * pp) * np.sqrt(2)
    alpha_flat = alpha_k.flatten()
    return alpha_flat


def W_at_alphas(xvec, yvec, w, alphas):
    return get_W_at_alphas(xvec*np.sqrt(2), yvec*np.sqrt(2), w, alphas)

    

def generate_alpha_batches(xvec, yvec, num_batches):
    #Randomly sample alphas to make multiple batches
    dim = len(xvec)
    xx, pp = np.meshgrid(xvec, yvec, indexing='xy') 

    alpha_k = (xx + 1j * pp) * np.sqrt(2)
    alpha_flat = alpha_k.flatten()
    
    np.random.shuffle(alpha_flat)

    total_points = len(alpha_flat)
    points_per_batch = total_points // num_batches
    usable_points = points_per_batch * num_batches #cut off any remainders from the division, so all batches are the same size

    alpha_trimmed = alpha_flat[:usable_points]
    batches = np.split(alpha_trimmed, num_batches)
    return np.array(batches)
        

def sample_wigner_spread(state, num_points=1000, num_samples=4):
    values = alpha_gen(state)
    N = len(values) #Gets amount of data points of wigner function
    samples = []

    if num_points >= N:
        segment = values[0:N]
        samples.append(segment)
        return samples

    # Calculate equally spread starting points (maximally separated)
    step = N//num_points
    offset = np.linspace(1, step, num_samples, dtype=int)

    for s in offset:
        segment = values[s::step]
        samples.append(segment)

    return samples