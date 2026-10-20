# Schrödinger Solver

**A Python-based numerical solver for the one-dimensional time-independent Schrödinger equation.**

This project explores quantum mechanical systems through numerical computation, visualization, and an interactive graphical user interface.

## Overview

The Schrödinger Solver is designed to investigate the behavior of quantum particles under different potential energy functions.

The project combines numerical methods, scientific computing, and data visualization to study fundamental problems in quantum mechanics.

The governing equation is the one-dimensional time-independent Schrödinger equation:

\[
-\frac{\hbar^2}{2m}\frac{d^2\psi(x)}{dx^2}+V(x)\psi(x)=E\psi(x)
\]

where:

- \(\hbar\): Reduced Planck constant.
- \(m\): Particle mass.
- \(V(x)\): Potential energy function.
- \(\psi(x)\): Wavefunction.
- \(E\): Energy eigenvalue.

## Features

- Numerical analysis of one-dimensional quantum systems.
- Visualization of wavefunctions and potential energy profiles.
- Support for different quantum mechanical potentials.
- Interactive graphical interface (`gui.py`).
- Modular Python architecture.
- Unit tests for numerical and supporting components.

## Supported Quantum Systems

| System | Description |
|---|---|
| Infinite Square Well | Particle confined within an ideal infinite potential well. |
| Harmonic Oscillator | Quantum particle subjected to a quadratic potential. |
| Gaussian Barrier | Particle interacting with a localized Gaussian potential barrier. |
| Periodic Cosine Potential | Quantum particle subjected to a spatially periodic potential. |

## Project Structure

The project contains the following main components:

| File | Purpose |
|---|---|
| `main.py` | Main program entry point. |
| `gui.py` | Interactive graphical user interface. |
| `tise.py` | Numerical treatment of the Schrödinger equation. |
| `interpolator.py` | Numerical interpolation utilities. |
| `display.py` | Visualization and plotting functions. |
| `interface.py` | Interface-related functionality. |
| `units.py` | Physical units and conversion utilities. |
| `test/` | Unit tests for core components. |
| `pyproject.toml` | Python project configuration. |

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/JavierC21-edu/Schrodinger-Solver.git
cd Schrodinger-Solver
```

### 2. Install the project

If the package is configured for editable installation in `pyproject.toml`, use:

```bash
python -m pip install -e .
```

### 3. Run the application

To launch the graphical interface:

```bash
python gui.py
```

The main program can also be executed, if configured as an entry point:

```bash
python main.py
```

## Testing

Unit tests are included to verify the functionality of the numerical solver and its supporting modules.

With `pytest` installed, run:

```bash
python -m pytest test/
```

## Scientific Applications

This project provides a computational environment for studying:

- Quantum confinement.
- Energy eigenvalues and eigenfunctions.
- Wavefunction behavior under different potentials.
- Numerical methods for differential equations.
- Scientific visualization of quantum mechanical systems.

## Technologies

**Programming language:** Python

**Scientific computing:** Numerical algorithms and computational methods.

**Visualization:** Graphical representation of quantum mechanical systems.

**Software design:** Modular architecture with automated tests.

## Future Improvements

Potential extensions include:

- Time-dependent Schrödinger equation simulations.
- Quantum tunneling and transmission analysis.
- Additional numerical solvers.
- Support for custom potential functions.
- Interactive parameter exploration.

## Author

**JavierC21-edu**

Physics student interested in quantum mechanics, computational physics, and numerical simulation.

## License

License information will be added if a license is selected for the repository.
