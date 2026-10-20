
"""
Unit conversions and physical constants for Schrodinger1D.

Internal units:
    Length: Bohr (a0)
    Energy: Hartree (Ha)
    Mass: Electron mass (me)
"""

# Physical constants
BOHR_TO_ANGSTROM = 0.5291772109
HARTREE_TO_EV = 27.2113862459

# Atomic units
HBAR_AU = 1.0
ELECTRON_MASS_AU = 1.0


# Length conversions
def bohr_to_angstrom(length):
    """Convert length from Bohr to Angstrom."""
    return length * BOHR_TO_ANGSTROM


def angstrom_to_bohr(length):
    """Convert length from Angstrom to Bohr."""
    return length / BOHR_TO_ANGSTROM


# Energy conversions
def hartree_to_ev(energy):
    """Convert energy from Hartree to electronvolts."""
    return energy * HARTREE_TO_EV


def ev_to_hartree(energy):
    """Convert energy from electronvolts to Hartree."""
    return energy / HARTREE_TO_EV


# General conversion functions
def convert_length(value, from_unit, to_unit):
    """Convert between Bohr and Angstrom."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit == to_unit:
        return value

    if from_unit == "bohr" and to_unit == "angstrom":
        return bohr_to_angstrom(value)

    if from_unit == "angstrom" and to_unit == "bohr":
        return angstrom_to_bohr(value)

    raise ValueError("Unsupported length conversion.")


def convert_energy(value, from_unit, to_unit):
    """Convert between Hartree and electronvolts."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()

    if from_unit == to_unit:
        return value

    if from_unit == "hartree" and to_unit == "ev":
        return hartree_to_ev(value)

    if from_unit == "ev" and to_unit == "hartree":
        return ev_to_hartree(value)

    raise ValueError("Unsupported energy conversion.")


# Module tests
if __name__ == "__main__":

    print("Schrodinger1D - Unit Conversion Tests")
    print("-" * 40)

    length = 2.0
    energy = 0.5

    print("\nLength conversions:")
    print(
        f"{length} Angstrom = "
        f"{angstrom_to_bohr(length):.6f} Bohr"
    )
    print(
        f"{length} Bohr = "
        f"{bohr_to_angstrom(length):.6f} Angstrom"
    )

    print("\nEnergy conversions:")
    print(
        f"{energy} Hartree = "
        f"{hartree_to_ev(energy):.6f} eV"
    )
    print(
        f"{energy} eV = "
        f"{ev_to_hartree(energy):.6f} Hartree"
    )

    print("\nGeneral conversion tests:")
    print(convert_length(1.0, "angstrom", "bohr"))
    print(convert_energy(1.0, "hartree", "ev"))
