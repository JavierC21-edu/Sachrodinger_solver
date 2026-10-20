
"""
Schrodinger1D - Graphical User Interface

A Tkinter interface for solving the one-dimensional
time-independent Schrodinger equation.

Uses the existing numerical solver without modifying
the original project modules.
"""

import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
import pickle

import numpy as np

from packets.schrodinger1d.interface import run_simulation
from packets.schrodinger1d.display import plot_results
from packets.schrodinger1d.units import hartree_to_ev


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "examples" / "data"

POTENTIALS = {
    "Infinite Square Well": "infinite_well.pkl",
    "Harmonic Oscillator": "harmonic.pkl",
    "Gaussian Barrier": "gaussian_barrier.pkl",
    "Periodic Cosine": "periodic_cosine.pkl"
}


class SchrodingerGUI:
    """Graphical interface for the Schrodinger1D solver."""

    def __init__(self, root):
        self.root = root
        self.root.title("Schrodinger1D Solver")
        self.root.geometry("650x570")
        self.root.minsize(550, 500)

        self.potential_var = tk.StringVar(
            value="Harmonic Oscillator"
        )
        self.length_var = tk.StringVar(value="10.0")
        self.spacing_var = tk.StringVar(value="0.05")
        self.states_var = tk.StringVar(value="5")

        self.create_widgets()

    def create_widgets(self):
        """Create the graphical interface."""

        frame = ttk.Frame(self.root, padding=20)
        frame.pack(fill="both", expand=True)

        title = ttk.Label(
            frame,
            text="Schrodinger1D Solver",
            font=("Arial", 18, "bold")
        )
        title.pack(pady=(0, 5))

        subtitle = ttk.Label(
            frame,
            text="Time-Independent Schrodinger Equation",
            font=("Arial", 10)
        )
        subtitle.pack(pady=(0, 20))

        settings = ttk.LabelFrame(
            frame,
            text="Simulation Parameters",
            padding=15
        )
        settings.pack(fill="x")

        ttk.Label(
            settings,
            text="Quantum Potential:"
        ).grid(row=0, column=0, sticky="w", pady=7)

        potential_menu = ttk.Combobox(
            settings,
            textvariable=self.potential_var,
            values=list(POTENTIALS.keys()),
            state="readonly",
            width=28
        )
        potential_menu.grid(
            row=0, column=1, padx=10, pady=7
        )

        parameters = [
            ("Domain Length (Bohr):", self.length_var),
            ("Grid Spacing (Bohr):", self.spacing_var),
            ("Number of States:", self.states_var)
        ]

        for index, (label, variable) in enumerate(
            parameters, start=1
        ):
            ttk.Label(
                settings,
                text=label
            ).grid(
                row=index,
                column=0,
                sticky="w",
                pady=7
            )

            ttk.Entry(
                settings,
                textvariable=variable,
                width=20
            ).grid(
                row=index,
                column=1,
                padx=10,
                pady=7
            )

        run_button = ttk.Button(
            frame,
            text="Run Simulation",
            command=self.run
        )
        run_button.pack(pady=20)

        results_frame = ttk.LabelFrame(
            frame,
            text="Eigenenergy Results",
            padding=10
        )
        results_frame.pack(
            fill="both",
            expand=True
        )

        self.results_text = tk.Text(
            results_frame,
            height=10,
            font=("Consolas", 10),
            wrap="word"
        )
        self.results_text.pack(
            fill="both",
            expand=True
        )

        self.results_text.insert(
            "end",
            "Select a potential and run the simulation."
        )
        self.results_text.config(state="disabled")

        ttk.Label(
            frame,
            text="Atomic Units: Bohr | Hartree | Electron Mass",
            font=("Arial", 9)
        ).pack(pady=10)

    def run(self):
        """Run the numerical simulation."""

        try:
            L = float(self.length_var.get())
            h = float(self.spacing_var.get())
            num_states = int(self.states_var.get())

            if not np.isfinite(L) or L <= 0:
                raise ValueError(
                    "Domain length must be positive and finite."
                )

            if not np.isfinite(h) or h <= 0:
                raise ValueError(
                    "Grid spacing must be positive and finite."
                )

            if num_states < 1:
                raise ValueError(
                    "Number of states must be positive."
                )

            filename = POTENTIALS[
                self.potential_var.get()
            ]

            filepath = DATA_DIR / filename

            # Load trusted local potential data.
            with filepath.open("rb") as file:
                data = pickle.load(file)

            original_L = float(data["L"])

            if original_L <= 0:
                raise ValueError("Invalid source domain.")

            x = (
                np.asarray(data["x"], dtype=float)
                * L / original_L
            )

            output = run_simulation(
                x=x,
                potential=data["potential"],
                L=L,
                h=h,
                boundary=data.get(
                    "boundary", "dirichlet"
                ),
                method=data.get(
                    "method", "linear"
                ),
                mass=data.get("mass", 1.0),
                num_states=num_states,
                length_unit="bohr",
                energy_unit="hartree",
                show_plot=False
            )

            solution = output["solution"]
            grid = output["grid"]
            energies = solution["energies"]

            self.results_text.config(state="normal")
            self.results_text.delete("1.0", "end")

            self.results_text.insert(
                "end",
                f"Potential: {self.potential_var.get()}\n"
                f"Domain Length: {L:.4f} Bohr\n"
                f"Grid Spacing: {grid['h']:.6f} Bohr\n\n"
                "Eigenenergies:\n"
                "------------------------------------\n"
            )

            for index, energy in enumerate(energies):
                energy_ev = hartree_to_ev(energy)

                self.results_text.insert(
                    "end",
                    f"State {index}: "
                    f"{energy:.8f} Ha | "
                    f"{energy_ev:.6f} eV\n"
                )

            self.results_text.config(state="disabled")
            self.root.update_idletasks()

            # Display numerical results using the
            # existing visualization module.
            plot_results(
                result=solution,
                potential=grid["potential"],
                show=True
            )

        except (
            ValueError,
            OSError,
            KeyError,
            TypeError
        ) as error:
            messagebox.showerror(
                "Simulation Error",
                str(error)
            )


def main():
    root = tk.Tk()
    SchrodingerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
