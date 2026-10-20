
import numpy as np
import matplotlib.pyplot as plt

data = np.load(
    "examples/results/gaussian_barrier_solution.npz"
)

print("Available variables:")
print(data.files)

energies = data["energies_hartree"]
x = data["x_bohr"]
psi = data["wavefunctions"]
potential = data["potential_hartree"]

print("\nEigenenergies (Hartree):")
for n, energy in enumerate(energies):
    print(f"State {n}: {energy:.8f}")

plt.figure(figsize=(10, 5))

plt.plot(x, potential, label="Potential V(x)")

for n in range(len(energies)):
    plt.plot(
        x,
        energies[n] + 0.05 * np.abs(psi[:, n])**2,
        label=f"State {n}"
    )

plt.xlabel("Position (Bohr)")
plt.ylabel("Energy (Hartree)")
plt.title("Gaussian Barrier - Schrodinger1D")
plt.legend()
plt.grid(True)
plt.show()
