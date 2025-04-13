import matplotlib.pyplot as plt


def plot_density_matrix(rho, title, ax_mag, ax_phase):
    abs_rho = np.abs(rho)
    phase_rho = np.angle(rho)

    im1 = ax_mag.imshow(abs_rho, cmap='viridis')

    ax_mag.set_title(f"{title} | Magnitude")

    return im1

plt.plot(sigma_list, fidelities, marker='o')
plt.xlabel("Noise level σ")
plt.ylabel("Fidelity F(ρ, ρ~)")
plt.title("Fidelity vs. Noise Level")
plt.grid(True)
plt.show()


fig, axs = plt.subplots(len(rho_with_noise_list), 2, figsize=(10, 4 * len(rho_with_noise_list)))

for i, (rho_noise, sigma) in enumerate(zip(rho_with_noise_list, sigma_list)):
    plot_density_matrix(
        rho_noise,
        f"Simulated Wigner → ρ (σ = {sigma})",
        axs[i, 0],
        axs[i, 0]  # same axes for magnitude/phase if simplified
    )
    plot_density_matrix(
        rho,
        "Experimental Wigner → ρ",
        axs[i, 1],
        axs[i, 1]
    )

# Optional: remove ticks
for row in axs:
    for ax in row:
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.show()
