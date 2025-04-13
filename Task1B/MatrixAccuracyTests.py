import numpy as np
import scipy as sc

def fidelity(rho_true, rho_est):
    """Compute the fidelity between two density matrices, rho_true and rho_est."""
    sqrt_rho = sc.linalg.sqrtm(rho_true)
    intermediate = sqrt_rho @ rho_est @ sqrt_rho
    sqrt_intermediate = sc.linalg.sqrtm(intermediate)
    return np.real(np.trace(sqrt_intermediate))*2


def trace_distance(rho_true, rho_est):
    """
    trace_distance function has two inputs, the two density matrices you plan on comparing. And then uses the formula for trace distance to compare
    the two density matrices, the lower the value the more similar the matrices.
    """ 
    eigvals = np.linalg.eigvalsh(rho_true - rho_est)        #Gets the eigenvalues of the difference in the density matrices
    trace_dist = 0.5 * np.sum(np.abs(eigvals))              #Gets the sum of the differences in the trace *.5 to get the Trace Distance
    return trace_dist