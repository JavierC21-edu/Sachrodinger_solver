# Schrodinger1D: A Numerical Solver for the One-Dimensional Schrödinger Equation

A Python package for solving the **one-dimensional time-independent Schrödinger equation (TISE)** using a finite-difference Hamiltonian. The project provides interpolation of sampled potentials, Dirichlet and periodic boundary conditions, numerical energy eigenvalues and wavefunctions, plots, and exportable example results.

## 1. Mathematical background

The stationary Schrödinger equation is

\[
\hat{H}\psi_n(x)=E_n\psi_n(x),\qquad
\hat{H}=-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}+V(x).
\]

Here, \(m\) is the particle mass, \(V(x)\) is the potential energy, and \(E_n\) and \(\psi_n\) are energy eigenvalues and stationary-state wavefunctions. The solver works internally with **atomic units**, for which \(\hbar=1\), the electron mass is \(1\), length is measured in Bohr, and energy in Hartree.

### Finite-difference discretization

On a uniform mesh with spacing \(h\), the second derivative is approximated by

\[
\frac{d^2\psi}{dx^2}\bigg|_{x_i}
\approx \frac{\psi_{i-1}-2\psi_i+\psi_{i+1}}{h^2}.
\]

Consequently, the Hamiltonian is represented by a sparse matrix,

\[
H=-\frac{1}{2m}D_2+\operatorname{diag}(V_1,\ldots,V_N),
\]

and the numerical problem becomes

\[
\mathbf{H}\boldsymbol{\psi}_n=E_n\boldsymbol{\psi}_n.
\]

The central-difference approximation has **second-order spatial accuracy** for sufficiently smooth solutions. The computed eigenvectors are normalized on the discrete mesh using

\[
h\sum_i |\psi_n(x_i)|^2=1.
\]

### Boundary conditions

- **Dirichlet:** \(\psi(0)=\psi(L)=0\). Only interior spatial nodes are unknown. This is appropriate for an infinite square well.
- **Periodic:** \(\psi(0)=\psi(L)\) and the discrete second-derivative matrix couples its first and last nodes. The periodic grid does not repeat the endpoint.

### Analytical verification: infinite square well

For \(V(x)=0\) inside \(0<x<L\) with infinite walls, the exact energies are

\[
E_n=\frac{n^2\pi^2\hbar^2}{2mL^2},\qquad n=1,2,\ldots
\]

In atomic units, \(\hbar=1\). Comparing this expression with computed eigenvalues provides a direct accuracy and grid-convergence check.

## 2. Features

- Sparse finite-difference Hamiltonian construction and eigensolution.
- Dirichlet and periodic boundary conditions.
- Interpolation of sampled potential data: linear, cubic spline, and PCHIP.
- Unit conversion between Bohr and Ångström, and between Hartree and electronvolts (eV).
- Visualization of potentials, energy levels, and wavefunctions.
- Reading example inputs stored as Python pickle files (`.pkl`).
- Export of eigenenergies (`.csv`), numerical solutions (`.npz`), and plots (`.png`).
- Unit tests for the solver, interpolation procedures, and unit conversions.

## 3. Project layout

```text
Schrödinger_solver/
├── main.py
├── pyproject.toml
├── README.md
├── packets/
│   └── schrodinger1d/
│       ├── __init__.py
│       ├── tise.py
│       ├── interpolator.py
│       ├── units.py
│       ├── display.py
│       └── interface.py
├── examples/
│   ├── data/
│   │   ├── generate_data.py
│   │   ├── infinite_well.pkl
│   │   ├── harmonic.pkl
│   │   ├── gaussian_barrier.pkl
│   │   └── periodic_cosine.pkl
│   └── results/
│       ├── generate_results.py
│       ├── *_energies.csv
│       ├── *_solution.npz
│       └── *_plots.png
└── test/
    ├── test_tise.py
    ├── test_interpolator.py
    └── test_units.py
```

The `packets/` directory is the **source-code directory** used by the setuptools configuration. The installed Python package is named `schrodinger1d`.

## 4. Requirements and installation

- Python **3.10 or newer** (Python 3.13 is supported by the project configuration).
- NumPy >= 1.24
- SciPy >= 1.10
- Matplotlib >= 3.7

Open a terminal in the **project root**, where `pyproject.toml` is located, and install the package in editable mode:

```bash
python -m pip install -e .
```

On Windows with Python 3.13, an alternative is:

```powershell
py -3.13 -m pip install -e .
```

## 5. Running simulations

The command-line interface accepts a named example or a path to a trusted pickle input file.

```bash
# Show help and examples
python main.py

# Infinite square well
python main.py --example well --L 10 --h 0.05

# Harmonic potential
python main.py --example harmonic --L 10 --h 0.05

# Gaussian barrier
python main.py --example gaussian --L 10 --h 0.05

# Periodic cosine potential
python main.py --example periodic --L 10 --h 0.05 --boundary periodic

# Load a potential from disk
python main.py --file examples/data/harmonic.pkl

# Compute energies without opening plots
python main.py --example well --no-plot
```

These commands assume that the corrected `interface.py` described in this project has been saved and the package installed. The equivalent installed command is `schrodinger1d` followed by the same options.

### Main command-line options

| Option | Purpose |
| --- | --- |
| `--example {well,harmonic,gaussian,periodic}` | Select a provided example. |
| `--file PATH` | Read a trusted `.pkl` potential file. |
| `--L VALUE` | Set the domain length (Bohr). |
| `--h VALUE` | Set the target spatial step (Bohr). |
| `--boundary {dirichlet,periodic}` | Choose the boundary condition. |
| `--states N` | Number of eigenstates to compute. |
| `--no-plot` | Disable graphical output. |

## 6. Example potentials and input format

The sample inputs use \(L=10\) Bohr, target spacing \(h=0.05\) Bohr, particle mass \(m=1\) in electron-mass units, and request five states.

| Input file | Potential \(V(x)\) (Hartree) | Boundary condition |
| --- | --- | --- |
| `infinite_well.pkl` | \(V(x)=0\) in the interior | Dirichlet |
| `harmonic.pkl` | \(0.02(x-L/2)^2\) | Dirichlet |
| `gaussian_barrier.pkl` | \(0.8\exp[-\tfrac12((x-L/2)/0.75)^2]\) | Dirichlet |
| `periodic_cosine.pkl` | \(0.15\cos(2\pi x/L)\) | Periodic |

**Important:** for the infinite square well, the infinite walls are represented by Dirichlet boundary conditions; the saved interior potential is zero rather than an array containing infinity.

Each pickle contains a dictionary similar to:

```python
{
    "x": x,                     # Original sampled positions, Bohr
    "potential": potential,     # Sampled potential, Hartree
    "L": 10.0,                  # Domain length, Bohr
    "h": 0.05,                  # Target numerical spacing, Bohr
    "boundary": "dirichlet",    # Or "periodic"
    "method": "cubic",          # "linear", "cubic", or "pchip"
    "mass": 1.0,                # Electron-mass units
    "num_states": 5,
    "length_unit": "bohr",
    "energy_unit": "hartree"
}
```

Sampled input coordinates need not have the same spacing as the final computational mesh: the interpolator constructs a uniform grid and evaluates the potential on that grid.

**Security note:** Python pickle deserialization can execute code. Load `.pkl` files only from trusted sources.

## 7. Generating inputs and saved results

From the project root, regenerate the four input files with:

```bash
python examples/data/generate_data.py
```

Generate numerical results and plots with:

```bash
python examples/results/generate_results.py
```

The results generator creates three output files per example, in `examples/results/`:

- `<name>_energies.csv`: state indices and energies in Hartree and eV.
- `<name>_solution.npz`: numerical positions, potential, eigenenergies, wavefunctions, spatial step, domain length, and boundary condition.
- `<name>_plots.png`: visual representation of the potential and calculated states.

For reproducibility, regenerate the data inputs before the results. Small floating-point differences in eigenvalues are possible across numerical libraries and computing environments; individual eigenfunctions may also differ by an overall sign without representing a different physical state.

## 8. Testing and numerical validation

Run the test suite from the project root:

```bash
python -m unittest discover -s test -p "test_*.py" -v
```

The tests are designed to examine:

1. Finite-difference matrices and Hamiltonian symmetry.
2. Infinite-square-well eigenenergies against the exact formula.
3. Wavefunction normalization, orthogonality, and eigenvalue-equation residuals.
4. Periodic boundary conditions and free-particle energies.
5. Convergence under grid refinement, including the expected second-order behavior.
6. Consistency of interpolated potentials and protection against malformed inputs.
7. Conversion factors, reversible unit conversions, and invalid units.

A successful test run supports the numerical implementation but does not prove that every potential or parameter choice is physically accurate. In particular, boundary effects, the domain size, and spatial resolution should be checked for each scientific application.

## 9. Numerical limitations

- This is a **one-dimensional stationary-state** solver; it does not calculate time evolution.
- The Hamiltonian is formed on a finite uniform grid, so discretization and finite-domain errors remain.
- Interpolation approximates potentials between sampled input points; insufficient input sampling may miss narrow features.
- The finite square well with finite exterior barriers is not the same problem as the infinite square well with Dirichlet walls.
- A periodic boundary condition models a periodic domain and should not be interpreted as a hard-wall confinement condition.
- The physical meaning of parameters and units should be checked before comparing results with experiments.

## 10. Project scope

The project is intended for **educational and computational physics applications**: studying the finite-difference discretization of a quantum Hamiltonian, understanding boundary conditions, and validating numerical eigenvalue problems against known solutions.

---

**Project:** Schrodinger1D  
**Version:** 0.1.0  
**Language:** Python  
**Numerical framework:** NumPy and SciPy  
**Visualization:** Matplotlib
