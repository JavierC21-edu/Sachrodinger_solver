
"""Command-line interface for the one-dimensional Schrodinger solver.

All numerical inputs use atomic units (Bohr, Hartree, electron masses).
Only load pickle files from trusted sources.
"""

import argparse
import pickle
from pathlib import Path

import numpy as np

from .interpolator import interpolate_for_tise
from .tise import solve_tise
from .display import plot_results


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "examples" / "data"


def load_pickle(path):
    """Load a trusted pickle potential and validate its required fields."""
    with Path(path).open("rb") as handle:
        data = pickle.load(handle)

    if not isinstance(data, dict):
        raise ValueError("The pickle file must contain a dictionary.")

    required = {"x", "potential", "L", "h"}
    missing = required - data.keys()

    if missing:
        raise ValueError(f"Missing fields: {', '.join(sorted(missing))}")

    return data


def run_simulation(
    x,
    potential,
    L,
    h,
    boundary="dirichlet",
    method="linear",
    mass=1.0,
    num_states=5,
    length_unit="bohr",
    energy_unit="hartree",
    show_plot=True,
):
    """Interpolate a potential and solve the 1D TISE."""

    if length_unit.lower() != "bohr":
        raise ValueError("Input positions must use Bohr.")

    if energy_unit.lower() != "hartree":
        raise ValueError("Input potential must use Hartree.")

    grid = interpolate_for_tise(
        x=x,
        potential=potential,
        L=L,
        h=h,
        boundary=boundary,
        method=method,
    )

    solution = solve_tise(
        potential=grid["potential"],
        h=grid["h"],
        boundary=boundary,
        mass=mass,
        num_states=num_states,
        L=grid["L"],
    )

    if show_plot:
        plot_results(
            result=solution,
            potential=grid["potential"],
            show=True,
        )

    return {
        "grid": grid,
        "solution": solution,
    }


def build_parser():
    """Build the command-line argument parser."""

    parser = argparse.ArgumentParser(
        description="Solve the 1D time-independent Schrodinger equation."
    )

    source = parser.add_mutually_exclusive_group(required=True)

    source.add_argument(
        "--file",
        type=Path,
        help="Path to a trusted pickle potential file",
    )

    source.add_argument(
        "--example",
        choices=["well", "harmonic", "gaussian", "periodic"],
        help="Built-in potential example",
    )

    parser.add_argument("--L", type=float)
    parser.add_argument("--h", type=float)

    parser.add_argument(
        "--boundary",
        choices=["dirichlet", "periodic"],
    )

    parser.add_argument("--states", type=int)

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Disable graphical visualization",
    )

    return parser


def main():
    """Run the solver from command-line arguments."""

    args = build_parser().parse_args()

    if args.file is not None:
        data = load_pickle(args.file)

    else:
        names = {
            "well": "infinite_well.pkl",
            "harmonic": "harmonic.pkl",
            "gaussian": "gaussian_barrier.pkl",
            "periodic": "periodic_cosine.pkl",
        }

        data = load_pickle(DATA_DIR / names[args.example])

    data = dict(data)

    if args.L is not None:
        original_L = float(data["L"])
        data["x"] = (
            np.asarray(data["x"], dtype=float)
            * args.L / original_L
        )
        data["L"] = args.L

    if args.h is not None:
        data["h"] = args.h

    if args.boundary is not None:
        data["boundary"] = args.boundary

    if args.states is not None:
        data["num_states"] = args.states

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
        show_plot=not args.no_plot,
    )

    print("\nEigenenergies (Hartree):")

    for index, energy in enumerate(
        output["solution"]["energies"]
    ):
        print(f"State {index}: {energy:.10f}")

    return output


if __name__ == "__main__":
    main()
