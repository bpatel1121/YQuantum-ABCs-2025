"AddGausNosie"
import numpy as np

def add_gaussian_noise(W, sigma):
    """
    Adds Gaussian noise with standard deviation sigma to the given Wigner data.
    """
    noise = np.random.normal(loc=0, scale=sigma, size=W.shape)
    return W + noise
