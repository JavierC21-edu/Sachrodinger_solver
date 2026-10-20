
"""
Schrodinger1D - Main Application

A Python-based package for solving the one-dimensional
time-independent Schrodinger equation.

Numerical method:
    Finite differences

Supported boundary conditions:
    - Dirichlet
    - Periodic

Features:
    - Potential interpolation
    - Atomic unit conversions
    - Eigenenergy calculations
    - Wavefunction calculations
    - Graphical visualization
    - Pickle file input
"""

import sys

from packets.schrodinger1d.interface import (
    main as run_interface,
    build_parser
)


def main():
    """Launch the Schrodinger1D application."""

    if len(sys.argv) == 1:

        print("=" * 55)
        print("           Schrodinger1D Solver")
        print("=" * 55)

        print("\nOne-dimensional time-independent")
        print("Schrodinger equation solver.\n")

        build_parser().print_help()

        print("\nExamples:")

        print(
            "\n1. Infinite square well:\n"
            "python main.py --example well --L 10 --h 0.05"
        )

        print(
            "\n2. Harmonic potential:\n"
            "python main.py --example harmonic --L 10 --h 0.05"
        )

        print(
            "\n3. Periodic boundary conditions:\n"
            "python main.py --example periodic --L 10 "
            "--h 0.05 --boundary periodic"
        )

        print(
            "\n4. Load potential from a pickle file:\n"
            "python main.py --file examples/data/harmonic.pkl"
        )

        return

    return run_interface()


if __name__ == "__main__":
    main()
