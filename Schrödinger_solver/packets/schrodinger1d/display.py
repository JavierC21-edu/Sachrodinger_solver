
"""
Graphical visualization for the Schrodinger1D package.

Available plots:
    - Potential and energy levels
    - Wavefunctions
    - Probability densities
    - Combined results

Internal units:
    Length: Bohr
    Energy: Hartree
"""

import numpy as np
import matplotlib.pyplot as plt

from .units import (
    BOHR_TO_ANGSTROM,
    bohr_to_angstrom,
    hartree_to_ev
)


def prepare_plot_data(
    x,
    potential,
    energies,
    length_unit="bohr",
    energy_unit="hartree"
):
    """
    Convert spatial and energy data for visualization.
    """

    x = np.asarray(x, dtype=float)
    potential = np.asarray(potential, dtype=float)
    energies = np.asarray(energies, dtype=float)

    length_unit = length_unit.lower()
    energy_unit = energy_unit.lower()

    if length_unit not in ("bohr", "angstrom"):
        raise ValueError("Invalid length unit.")

    if energy_unit not in ("hartree", "ev"):
        raise ValueError("Invalid energy unit.")

    if length_unit == "angstrom":
        x = bohr_to_angstrom(x)

    if energy_unit == "ev":
        potential = hartree_to_ev(potential)
        energies = hartree_to_ev(energies)

    return x, potential, energies


def prepare_wavefunctions(wavefunctions, length_unit="bohr"):
    """
    Convert wavefunction normalization to the selected
    spatial units.
    """

    psi = np.asarray(wavefunctions)

    if length_unit == "angstrom":
        psi = psi / np.sqrt(BOHR_TO_ANGSTROM)

    elif length_unit != "bohr":
        raise ValueError("Invalid length unit.")

    return psi


def validate_results(result):
    """
    Validate the result dictionary from solve_tise().
    """

    required = ("x", "energies", "wavefunctions")

    for key in required:
        if key not in result:
            raise ValueError(f"Missing result key: {key}")

    x = np.asarray(result["x"])
    energies = np.asarray(result["energies"])
    psi = np.asarray(result["wavefunctions"])

    if x.ndim != 1 or energies.ndim != 1:
        raise ValueError("x and energies must be 1D arrays.")

    if psi.shape != (len(x), len(energies)):
        raise ValueError("Invalid wavefunction dimensions.")

    if len(x) == 0 or len(energies) == 0:
        raise ValueError("Result arrays cannot be empty.")

    if (
        not np.all(np.isfinite(x))
        or not np.all(np.isfinite(energies))
        or not np.all(np.isfinite(psi))
    ):
        raise ValueError("Results must contain finite values.")


def plot_potential(
    result,
    potential,
    length_unit="bohr",
    energy_unit="hartree",
    scale=0.3,
    save_path=None,
    show=True
):
    """
    Plot potential, energy levels and shifted wavefunctions.

    Wavefunctions are visually scaled and shifted to
    their corresponding energy levels.
    """

    validate_results(result)

    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and nonnegative.")

    x, V, energies = prepare_plot_data(
        result["x"],
        potential,
        result["energies"],
        length_unit,
        energy_unit
    )

    if V.shape != x.shape or not np.all(np.isfinite(V)):
        raise ValueError("Invalid potential array.")

    psi = prepare_wavefunctions(
        result["wavefunctions"],
        length_unit
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, V, color="black", linewidth=2, label="V(x)")

    if len(energies) > 1:
        gaps = np.diff(np.sort(energies))
        positive_gaps = gaps[gaps > 0]
    else:
        positive_gaps = np.array([])

    if len(positive_gaps):
        energy_scale = np.min(positive_gaps)
    else:
        energy_scale = max(np.ptp(V), 1.0)

    for n, energy in enumerate(energies):

        ax.axhline(
            energy,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.6
        )

        state = np.real(psi[:, n])
        amplitude = np.max(np.abs(state))

        if amplitude > 0:
            state = state / amplitude

        shifted = energy + scale * energy_scale * state

        ax.plot(
            x,
            shifted,
            label=f"State {n}: E = {energy:.4f}"
        )

    x_label = "Position (Å)" if length_unit == "angstrom" else "Position (Bohr)"
    e_label = "Energy (eV)" if energy_unit == "ev" else "Energy (Ha)"

    ax.set_xlabel(x_label)
    ax.set_ylabel(e_label)
    ax.set_title("Potential, Energy Levels and Wavefunctions")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_wavefunctions(
    result,
    length_unit="bohr",
    save_path=None,
    show=True
):
    """
    Plot normalized eigenfunctions.
    """

    validate_results(result)

    x, _, _ = prepare_plot_data(
        result["x"],
        np.zeros_like(result["x"]),
        result["energies"],
        length_unit,
        "hartree"
    )

    psi = prepare_wavefunctions(
        result["wavefunctions"],
        length_unit
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for n in range(psi.shape[1]):

        ax.plot(
            x,
            np.real(psi[:, n]),
            label=f"State {n}"
        )

    x_label = "Position (Å)" if length_unit == "angstrom" else "Position (Bohr)"
    psi_unit = "Å⁻¹/²" if length_unit == "angstrom" else "Bohr⁻¹/²"

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Wavefunction ({psi_unit})")
    ax.set_title("Quantum Wavefunctions")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_probability_density(
    result,
    length_unit="bohr",
    save_path=None,
    show=True
):
    """
    Plot probability densities |psi(x)|^2.
    """

    validate_results(result)

    x, _, _ = prepare_plot_data(
        result["x"],
        np.zeros_like(result["x"]),
        result["energies"],
        length_unit,
        "hartree"
    )

    psi = prepare_wavefunctions(
        result["wavefunctions"],
        length_unit
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for n in range(psi.shape[1]):

        probability = np.abs(psi[:, n])**2

        ax.plot(
            x,
            probability,
            label=f"State {n}"
        )

    x_label = "Position (Å)" if length_unit == "angstrom" else "Position (Bohr)"
    density_unit = "Å⁻¹" if length_unit == "angstrom" else "Bohr⁻¹"

    ax.set_xlabel(x_label)
    ax.set_ylabel(f"Probability Density ({density_unit})")
    ax.set_title("Probability Densities")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def plot_results(
    result,
    potential,
    length_unit="bohr",
    energy_unit="hartree",
    save_path=None,
    show=True
):
    """
    Create a combined figure containing:
        1. Potential and energy levels
        2. Wavefunctions
        3. Probability densities
    """

    validate_results(result)

    x, V, energies = prepare_plot_data(
        result["x"],
        potential,
        result["energies"],
        length_unit,
        energy_unit
    )

    if V.shape != x.shape or not np.all(np.isfinite(V)):
        raise ValueError("Invalid potential array.")

    psi = prepare_wavefunctions(
        result["wavefunctions"],
        length_unit
    )

    fig, axes = plt.subplots(
        3, 1,
        figsize=(10, 12),
        sharex=True
    )

    # Potential and energy levels

    axes[0].plot(
        x, V,
        color="black",
        linewidth=2,
        label="V(x)"
    )

    for n, energy in enumerate(energies):

        axes[0].axhline(
            energy,
            linestyle="--",
            label=f"E{n} = {energy:.4f}"
        )

    axes[0].set_title("Potential and Energy Levels")
    axes[0].set_ylabel(
        "Energy (eV)" if energy_unit == "ev"
        else "Energy (Ha)"
    )
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Wavefunctions

    for n in range(psi.shape[1]):

        axes[1].plot(
            x,
            np.real(psi[:, n]),
            label=f"State {n}"
        )

    axes[1].set_title("Wavefunctions")
    axes[1].set_ylabel("Wavefunction")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Probability densities

    for n in range(psi.shape[1]):

        axes[2].plot(
            x,
            np.abs(psi[:, n])**2,
            label=f"State {n}"
        )

    axes[2].set_title("Probability Densities")
    axes[2].set_ylabel("Probability Density")
    axes[2].set_xlabel(
        "Position (Å)" if length_unit == "angstrom"
        else "Position (Bohr)"
    )
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig, axes
