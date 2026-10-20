
"""
Schrodinger1D
=============

A Python package for solving the one-dimensional
time-independent Schrodinger equation.

Features
--------
- Atomic unit conversions
- Finite-difference Hamiltonian construction
- Periodic and Dirichlet boundary conditions
- Numerical eigenvalue and eigenfunction calculation
- Potential interpolation
- Graphical visualization
- Command-line and Python interfaces
"""

__version__ = "0.1.0"
__author__ = "Schrodinger1D Project"


# Physical constants and unit conversions
from .units import (
    BOHR_TO_ANGSTROM,
    HARTREE_TO_EV,
    HBAR_AU,
    ELECTRON_MASS_AU,
    bohr_to_angstrom,
    angstrom_to_bohr,
    hartree_to_ev,
    ev_to_hartree,
    convert_length,
    convert_energy
)


# Time-independent Schrodinger equation solver
from .tise import (
    second_derivative_matrix,
    build_hamiltonian,
    solve_tise
)


# Potential interpolation
from .interpolator import (
    create_grid,
    interpolate_potential,
    interpolate_for_tise
)


# Graphical visualization
from .display import (
    plot_potential,
    plot_wavefunctions,
    plot_probability_density,
    plot_results
)


# High-level simulation interface
from .interface import (
    load_pickle,
    run_simulation
)


# Public package interface
__all__ = [
    "BOHR_TO_ANGSTROM",
    "HARTREE_TO_EV",
    "HBAR_AU",
    "ELECTRON_MASS_AU",
    "bohr_to_angstrom",
    "angstrom_to_bohr",
    "hartree_to_ev",
    "ev_to_hartree",
    "convert_length",
    "convert_energy",
    "second_derivative_matrix",
    "build_hamiltonian",
    "solve_tise",
    "create_grid",
    "interpolate_potential",
    "interpolate_for_tise",
    "plot_potential",
    "plot_wavefunctions",
    "plot_probability_density",
    "plot_results",
    "load_pickle",
    "run_simulation"
]
