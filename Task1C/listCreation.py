def fidelity(rho_true, rho_est):
    """Compute the fidelity between two density matrices, rho_true and rho_est."""
    sqrt_rho = sc.linalg.sqrtm(rho_true)
    intermediate = sqrt_rho @ rho_est @ sqrt_rho
    sqrt_intermediate = sc.linalg.sqrtm(intermediate)
    return np.real(np.trace(sqrt_intermediate))**2

rho = density_from_wigner(xvec, yvec, w, alpha0=0, num_points=2000, num_batches=20, radius=1.2, N=10, fock=False) #convert raw data into this density matrix

sigma_list = [0.05,.1,.5,1] #choose the arbitrary sigmas
rho_with_noise_list = []    #initial list when noise(sigma) is applied

for sigmas in sigma_list: #apply gaussian noise to w and create list of new w's with different noise applied(per different sigmas)
    w_noise = add_gaussian_noise(w, sigmas)

    rho_with_noise = density_from_wigner(xvec, yvec, w_noise, alpha0=0, num_points=2000, num_batches=20, radius=1.2, N=10, fock=False)
    rho_with_noise_list.append(rho_with_noise) #create the rho with noise lists

fidelities = [] 
for rho_reconstructed in rho_with_noise_list: #create the list of fidelity values, for each w with noise applied
    f = fidelity(rho, rho_reconstructed)
    fidelities.append(f)

