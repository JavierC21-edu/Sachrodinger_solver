
"""
Potential interpolation for the Schrodinger1D package.

Available methods:
    - Linear interpolation
    - Cubic spline interpolation
    - PCHIP interpolation

Supported boundary conditions:
    - Dirichlet (non-periodic)
    - Periodic

Internal units:
    Length: Bohr
    Potential energy: Hartree
"""

import numpy as np

from scipy.interpolate import (
    interp1d,
    CubicSpline,
    PchipInterpolator
)


def create_grid(
    L,
    h,
    boundary="dirichlet",
    adjust_spacing=False
):
    """
    Generate a spatial grid compatible with solve_tise().

    Parameters
    ----------
    L : float
        Domain length in Bohr.
    h : float
        Requested spacing in Bohr.
    boundary : str
        'dirichlet' or 'periodic'.
    adjust_spacing : bool
        Allow adjusting h to divide L exactly.

    Returns
    -------
    dict
        x : ndarray
            Computational grid.
        h : float
            Actual grid spacing.
        L : float
            Domain length.
        boundary : str
            Boundary condition.
    """

    if not np.isfinite(L) or L <= 0:
        raise ValueError("L must be positive and finite.")

    if not np.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite.")

    if boundary not in ("dirichlet", "periodic"):
        raise ValueError("Invalid boundary condition.")

    ratio = L / h

    intervals = int(np.rint(ratio))

    if intervals < 4:
        raise ValueError(
            "At least four spatial intervals are required."
        )

    if not adjust_spacing:

        if not np.isclose(
            ratio,
            intervals,
            rtol=1e-9,
            atol=1e-9
        ):
            raise ValueError(
                "L/h must be an integer. "
                "Choose another h or enable adjust_spacing."
            )

    h_actual = L / intervals

    if boundary == "dirichlet":

        x = np.arange(1, intervals) * h_actual

    else:

        x = np.arange(intervals) * h_actual

    return {
        "x": x,
        "h": h_actual,
        "L": L,
        "boundary": boundary
    }


def validate_potential_data(x, potential):
    """
    Validate spatial positions and potential values.

    Returns
    -------
    tuple
        Validated position and potential arrays.
    """

    x = np.asarray(x, dtype=float)
    potential = np.asarray(potential, dtype=float)

    if x.ndim != 1 or potential.ndim != 1:
        raise ValueError(
            "Position and potential must be 1D arrays."
        )

    if len(x) != len(potential):
        raise ValueError(
            "Position and potential must have equal lengths."
        )

    if len(x) < 2:
        raise ValueError(
            "At least two potential samples are required."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "Position values must be finite."
        )

    if not np.all(np.isfinite(potential)):
        raise ValueError(
            "Potential values must be finite."
        )

    if not np.all(np.diff(x) > 0):
        raise ValueError(
            "Position values must be strictly increasing."
        )

    return x, potential


def interpolate_potential(
    x,
    potential,
    x_new,
    method="linear",
    periodic=False
):
    """
    Interpolate a potential onto a new spatial grid.

    Parameters
    ----------
    x : array_like
        Original positions in Bohr.
    potential : array_like
        Original potential values in Hartree.
    x_new : array_like
        Target positions in Bohr.
    method : str
        'linear', 'cubic', or 'pchip'.
    periodic : bool
        Use a periodic cubic spline when applicable.
        For periodic data, endpoints must satisfy
        V(0) = V(L).

    Returns
    -------
    ndarray
        Interpolated potential values in Hartree.
    """

    x, potential = validate_potential_data(
        x, potential
    )

    x_new = np.asarray(x_new, dtype=float)

    if x_new.ndim != 1:
        raise ValueError("Target grid must be 1D.")

    if not np.all(np.isfinite(x_new)):
        raise ValueError(
            "Target grid values must be finite."
        )

    if len(x_new) == 0:
        raise ValueError("Target grid cannot be empty.")

    if (
        np.any(x_new < x[0] - 1e-10)
        or np.any(x_new > x[-1] + 1e-10)
    ):
        raise ValueError(
            "Target grid exceeds the original domain."
        )

    method = method.lower()

    if method not in ("linear", "cubic", "pchip"):
        raise ValueError(
            "Method must be linear, cubic or pchip."
        )

    if method == "linear":

        interpolator = interp1d(
            x,
            potential,
            kind="linear",
            bounds_error=True
        )

    elif method == "cubic":

        if len(x) < 3:
            raise ValueError(
                "Cubic interpolation requires at least 3 points."
            )

        if periodic:

            if not np.isclose(
                potential[0],
                potential[-1],
                rtol=1e-9,
                atol=1e-12
            ):
                raise ValueError(
                    "Periodic interpolation requires "
                    "matching endpoint values."
                )

            interpolator = CubicSpline(
                x,
                potential,
                bc_type="periodic"
            )

        else:

            interpolator = CubicSpline(
                x,
                potential
            )

    else:

        interpolator = PchipInterpolator(
            x,
            potential
        )

    return np.asarray(
        interpolator(x_new),
        dtype=float
    )


def interpolate_for_tise(
    x,
    potential,
    L,
    h,
    boundary="dirichlet",
    method="linear",
    adjust_spacing=False
):
    """
    Interpolate tabulated potential data for solve_tise().

    The input data must cover the full domain [0, L].

    For periodic boundaries, input samples may omit
    the endpoint L. It is reconstructed from V(0).

    Parameters
    ----------
    x : array_like
        Original positions in Bohr.
    potential : array_like
        Potential values in Hartree.
    L : float
        Domain length in Bohr.
    h : float
        Requested grid spacing in Bohr.
    boundary : str
        'dirichlet' or 'periodic'.
    method : str
        'linear', 'cubic', or 'pchip'.
    adjust_spacing : bool
        Allow a small adjustment of grid spacing.

    Returns
    -------
    dict
        x : ndarray
            New computational grid.
        potential : ndarray
            Interpolated potential in Hartree.
        h : float
            Actual grid spacing.
        L : float
            Domain length.
        boundary : str
            Boundary condition.
        method : str
            Interpolation method.
    """

    x, potential = validate_potential_data(
        x, potential
    )

    grid = create_grid(
        L=L,
        h=h,
        boundary=boundary,
        adjust_spacing=adjust_spacing
    )

    if not np.isclose(
        x[0],
        0.0,
        rtol=0,
        atol=1e-10
    ):
        raise ValueError(
            "Input positions must start at x = 0."
        )

    if np.any(x < -1e-10) or np.any(x > L + 1e-10):
        raise ValueError(
            "Input positions must lie inside [0, L]."
        )

    # Work with a copy of the input arrays.
    x_data = x.copy()
    V_data = potential.copy()

    if boundary == "dirichlet":

        if not np.isclose(
            x_data[-1],
            L,
            rtol=1e-9,
            atol=1e-10
        ):
            raise ValueError(
                "Dirichlet input must include x = L."
            )

    else:

        # The periodic endpoint is identical to x = 0.
        if np.isclose(
            x_data[-1],
            L,
            rtol=1e-9,
            atol=1e-10
        ):

            if not np.isclose(
                V_data[0],
                V_data[-1],
                rtol=1e-9,
                atol=1e-12
            ):
                raise ValueError(
                    "Periodic potential requires V(0) = V(L)."
                )

            V_data[-1] = V_data[0]

        else:

            x_data = np.append(x_data, L)
            V_data = np.append(V_data, V_data[0])

    # Enforce exact computational endpoints.
    x_data[0] = 0.0
    x_data[-1] = L

    V_new = interpolate_potential(
        x=x_data,
        potential=V_data,
        x_new=grid["x"],
        method=method,
        periodic=(boundary == "periodic")
    )

    return {
        "x": grid["x"],
        "potential": V_new,
        "h": grid["h"],
        "L": grid["L"],
        "boundary": boundary,
        "method": method
    }


if __name__ == "__main__":

    print("Schrodinger1D - Interpolator Test")
    print("-" * 45)

    # Original coarse grid.
    L = 10.0

    x_original = np.linspace(0, L, 11)

    # Harmonic potential in atomic units.
    V_original = 0.02 * (x_original - L / 2)**2

    # Interpolation onto a finer grid.
    result = interpolate_for_tise(
        x=x_original,
        potential=V_original,
        L=L,
        h=0.05,
        boundary="dirichlet",
        method="cubic"
    )

    print(f"Domain length: {result['L']:.4f} Bohr")
    print(f"Grid spacing: {result['h']:.4f} Bohr")

    print(
        f"Original samples: {len(x_original)}"
    )

    print(
        f"Interpolated samples: {len(result['x'])}"
    )

    print("\nFirst five interpolated values:")

    for position, value in zip(
        result["x"][:5],
        result["potential"][:5]
    ):
        print(
            f"x = {position:.4f} Bohr, "
            f"V = {value:.6f} Ha"
        )
