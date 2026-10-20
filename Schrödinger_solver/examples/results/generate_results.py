"""Generate saved results from the Schrodinger1D package.

Run from the project root after `pip install -e .`:
    python examples/results/generate_results.py

Only read trusted pickle files.
"""
from pathlib import Path
import sys
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from packets.schrodinger1d.interface import run_simulation
from packets.schrodinger1d.display import plot_results
from packets.schrodinger1d.units import hartree_to_ev

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
NAMES = ("infinite_well", "harmonic", "gaussian_barrier", "periodic_cosine")

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name in NAMES:
        with (DATA_DIR / f"{name}.pkl").open("rb") as handle:
            data = pickle.load(handle)  # Trusted local files only.

        output = run_simulation(
            x=data["x"],
            potential=data["potential"],
            L=data["L"],
            h=data["h"],
            boundary=data.get("boundary", "dirichlet"),
            method=data.get("method", "linear"),
            mass=data.get("mass", 1.0),
            num_states=data.get("num_states", 5),
            length_unit=data.get("length_unit", "bohr"),
            energy_unit=data.get("energy_unit", "hartree"),
            show_plot=False
        )
        grid = output["grid"]
        solution = output["solution"]
        energies = solution["energies"]

        # Save reproducible numerical output.
        np.savez(
            RESULTS_DIR / f"{name}_solution.npz",
            x_bohr=solution["x"],
            potential_hartree=grid["potential"],
            energies_hartree=energies,
            wavefunctions=solution["wavefunctions"],
            h_bohr=solution["h"],
            L_bohr=solution["L"],
            boundary=solution["boundary"]
        )
        with (RESULTS_DIR / f"{name}_energies.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(["state_index", "energy_hartree", "energy_eV"])
            for index, value in enumerate(energies):
                writer.writerow([index, f"{value:.12g}", f"{hartree_to_ev(value):.12g}"])

        figure, _ = plot_results(
            result=solution, potential=grid["potential"],
            length_unit="angstrom", energy_unit="ev",
            save_path=RESULTS_DIR / f"{name}_plots.png", show=False
        )
        plt.close(figure)
        print(f"Saved results for {name}")

if __name__ == "__main__":
    main()
