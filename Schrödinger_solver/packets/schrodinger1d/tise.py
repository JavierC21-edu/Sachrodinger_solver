
"""
One-dimensional time-independent Schrodinger equation solver.

Numerical method:
    Second-order central finite differences.

Boundary conditions:
    - Dirichlet (non-periodic)
    - Periodic

Internal atomic units:
    Length: Bohr
    Energy: Hartree
    Mass: Electron mass
    Reduced Planck constant: hbar = 1
"""

import numpy as np

from scipy.linalg import eigh
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh


def second_derivative_matrix(n, h, boundary="dirichlet"):
    """
    Construct the second-derivative matrix.

    Parameters
    ----------
    n : int
        Number of grid points.
    h : float
        Grid spacing in Bohr.
    boundary : str
        'dirichlet' or 'periodic'.

    Returns
    -------
    scipy.sparse.csr_matrix
        Discretized second-derivative operator.
    """

    if not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)):
        raise ValueError("n must be an integer.")

    if n < 3:
        raise ValueError("At least 3 grid points are required.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("Grid spacing must be positive and finite.")

    if boundary not in ("dirichlet", "periodic"):
        raise ValueError("Invalid boundary condition.")

    main = -2.0 * np.ones(n)
    off = np.ones(n - 1)

    D2 = diags(
        [off, main, off],
        offsets=[-1, 0, 1],
        format="csr"
    )

    D2 = D2 / h**2

    if boundary == "periodic":

        D2 = D2.tolil()

        D2[0, -1] = 1.0 / h**2
        D2[-1, 0] = 1.0 / h**2

        D2 = D2.tocsr()

    return D2


def build_hamiltonian(
    potential,
    h,
    boundary="dirichlet",
    mass=1.0
):
    """
    Construct the Hamiltonian matrix.

    H = -(1 / 2m) * D2 + V

    Parameters
    ----------
    potential : array_like
        Potential values in Hartree.
    h : float
        Grid spacing in Bohr.
    boundary : str
        Boundary condition.
    mass : float
        Particle mass in electron-mass units.

    Returns
    -------
    scipy.sparse.csr_matrix
        Hamiltonian matrix.
    """

    potential = np.asarray(potential, dtype=float)

    if potential.ndim != 1:
        raise ValueError("Potential must be a 1D array.")

    if not np.all(np.isfinite(potential)):
        raise ValueError("Potential values must be finite.")

    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Mass must be positive and finite.")

    n = len(potential)

    D2 = second_derivative_matrix(n, h, boundary)

    V = diags(potential, 0, format="csr")

    H = -(1.0 / (2.0 * mass)) * D2 + V

    return H


def solve_tise(
    potential,
    h,
    boundary="dirichlet",
    mass=1.0,
    num_states=5,
    L=None
):
    """
    Solve the 1D time-independent Schrodinger equation.

    Parameters
    ----------
    potential : array_like
        Potential values in Hartree.
        For Dirichlet boundaries, provide interior points.
        For periodic boundaries, do not duplicate x = L.
    h : float
        Grid spacing in Bohr.
    boundary : str
        'dirichlet' or 'periodic'.
    mass : float
        Particle mass in electron-mass units.
    num_states : int
        Number of eigenstates to calculate.
    L : float, optional
        Domain length in Bohr. Must match the grid.

    Returns
    -------
    dict
        x : ndarray
            Spatial grid in Bohr.
        energies : ndarray
            Eigenenergies in Hartree.
        wavefunctions : ndarray
            Normalized eigenfunctions stored in columns.
        h : float
            Grid spacing in Bohr.
        L : float
            Domain length in Bohr.
        boundary : str
            Applied boundary condition.
    """

    potential = np.asarray(potential, dtype=float)

    if potential.ndim != 1:
        raise ValueError("Potential must be a 1D array.")

    n = len(potential)

    if not isinstance(num_states, (int, np.integer)) or isinstance(
        num_states, (bool, np.bool_)
    ):
        raise ValueError("num_states must be an integer.")

    if not 1 <= num_states <= n:
        raise ValueError(
            "num_states must be between 1 and the grid size."
        )

    H = build_hamiltonian(
        potential=potential,
        h=h,
        boundary=boundary,
        mass=mass
    )

    # Construct the spatial grid.

    if boundary == "dirichlet":

        expected_L = (n + 1) * h
        x = np.arange(1, n + 1) * h

    else:

        expected_L = n * h
        x = np.arange(n) * h

    if L is not None:

        if not np.isfinite(L) or L <= 0:
            raise ValueError("L must be positive and finite.")

        if not np.isclose(L, expected_L, rtol=1e-8, atol=1e-12):
            raise ValueError(
                "L is inconsistent with the grid spacing "
                "and boundary condition."
            )

    else:

        L = expected_L

    # Solve the eigenvalue problem.
    # Dense diagonalization is suitable for small matrices
    # or when all states are requested.

    if n <= 256 or num_states >= n - 1:

        energies, wavefunctions = eigh(H.toarray())

        energies = energies[:num_states]
        wavefunctions = wavefunctions[:, :num_states]

    else:

        energies, wavefunctions = eigsh(
            H,
            k=num_states,
            which="SA"
        )

        order = np.argsort(energies)

        energies = energies[order]
        wavefunctions = wavefunctions[:, order]

    # Normalize wavefunctions on the spatial grid.

    wavefunctions = wavefunctions / np.sqrt(h)

    return {
        "x": x,
        "energies": energies,
        "wavefunctions": wavefunctions,
        "h": h,
        "L": L,
        "boundary": boundary
    }


# Numerical verification
if __name__ == "__main__":

    print("Schrodinger1D - TISE Solver Test")
    print("-" * 45)

    # Infinite square well in atomic units.
    L = 10.0
    n = 200
    h = L / (n + 1)

    potential = np.zeros(n)

    result = solve_tise(
        potential=potential,
        h=h,
        boundary="dirichlet",
        mass=1.0,
        num_states=5,
        L=L
    )

    numerical_energies = result["energies"]

    quantum_numbers = np.arange(1, 6)

    analytical_energies = (
        quantum_numbers**2 * np.pi**2
        / (2.0 * L**2)
    )

    print("\nInfinite Square Well")
    print(f"Domain length: {L:.4f} Bohr")
    print(f"Grid spacing: {h:.6f} Bohr")

    print("\nState | Numerical | Analytical | Error")
    print("-" * 45)

    for i in range(5):

        error = abs(
            numerical_energies[i]
            - analytical_energies[i]
        )

        print(
            f"{i + 1:5d} | "
            f"{numerical_energies[i]:9.6f} | "
            f"{analytical_energies[i]:10.6f} | "
            f"{error:.6e}"
        )
