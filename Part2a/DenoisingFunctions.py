import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def comp_dub_int_Ws(W, dx, dp): 
    return np.sum(W)*dx*dp

def estimate_background(W_measured, edge_width=5):
    """
    Estimate background offset b using the edges of the phase space.
    edge_width sets the number of pixels from the borders to include.
    """
    nx, np_points = W_measured.shape
    # Collect boundary pixels
    top = W_measured[:edge_width, :]
    bottom = W_measured[-edge_width:, :]
    left = W_measured[:, :edge_width]
    right = W_measured[:, -edge_width:]
    
    # Average over all edge pixels
    b_est = np.mean(np.concatenate((top.flatten(), bottom.flatten(), left.flatten(), right.flatten())))
    return b_est

def estimate_scale(W_measured, b_est, dx, dp):
    """
    Estimate the scaling parameter a based on the normalization constraint.
    dx, dp: spacing in x and p dimensions
    """
    # Subtract background noise
    corrected = W_measured - b_est
    
    integral = comp_dub_int_Ws(W_measured,dx,dp)

    a_est = integral 

    return a_est

def affine_correct(W_measured, a_est, b_est):
    """
    Apply the affine correction to get an estimate of the true Wigner function.
    """
    W_corrected = (W_measured - b_est) / a_est #Applying given equation
    return W_corrected

def process_wigner(W_measured, dx, dp, sigma=.5):
    """
    Process one Wigner function:
    - Estimate affine parameters (a and b)
    - Correct the affine distortion
    - Apply Gaussian filtering (if sigma > 0)
    Returns the corrected (raw) and denoised versions.
    """

    # Estimate background offset from edges
    b_est = estimate_background(W_measured)
    
    # Estimate scaling factor using normalization
    a_est = estimate_scale(W_measured, b_est, dx, dp)
    
    # Correct for affine distortion
    W_corrected = affine_correct(W_measured, a_est, b_est)
    
    # Copy raw corrected for benchmarking
    W_raw = W_corrected.copy()
    
    W_denoised = gaussian_filter(W_corrected, sigma=sigma)
    
    return W_denoised