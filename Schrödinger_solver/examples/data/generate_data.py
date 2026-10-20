"""Generate trusted sample potential files for Schrodinger1D.

All coordinates use Bohr, all potentials use Hartree, and
h is the requested fine-grid spacing (not the sample spacing).
"""

import pickle
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).resolve().parent
L = 10.0
H_TARGET = 0.05
X = np.linspace(0.0, L, 21)  # 21 original samples, spacing 0.5 Bohr


def save_potential(filename, potential, boundary="dirichlet", method="cubic"):
    """Write a dictionary understood by schrodinger1d.interface.load_pickle."""
    data = {
        "x": X,
        "potential": np.asarray(potential, dtype=float),
        "L": L,
        "h": H_TARGET,
        "boundary": boundary,
        "method": method,
        "mass": 1.0,
        "num_states": 5,
        "length_unit": "bohr",
        "energy_unit": "hartree",
    }

    path = DATA_DIR / filename
    with path.open("wb") as stream:
        pickle.dump(data, stream, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Created: {path.name}")


def main():
    save_potential("infinite_well.pkl", np.zeros_like(X), method="linear")
    save_potential("harmonic.pkl", 0.02 * (X - L / 2.0) ** 2)
    save_potential(
        "gaussian_barrier.pkl",
        0.8 * np.exp(-0.5 * ((X - L / 2.0) / 0.75) ** 2),
    )
    save_potential(
        "periodic_cosine.pkl",
        0.15 * np.cos(2.0 * np.pi * X / L),
        boundary="periodic",
    )


if __name__ == "__main__":
    main()
