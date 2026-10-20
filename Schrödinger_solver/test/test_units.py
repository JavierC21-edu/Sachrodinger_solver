
"""
Unit tests for physical constants and conversions
in the Schrodinger1D package.

Tests:
    - Atomic unit constants
    - Length conversions
    - Energy conversions
    - Round-trip conversions
    - NumPy array compatibility
    - General conversion functions
    - Invalid unit handling
"""

import unittest
import numpy as np

from packets.schrodinger1d.units import (
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


class TestUnits(unittest.TestCase):
    """Test atomic units and unit conversions."""

    def test_physical_constants(self):
        """Verify the atomic unit constants."""

        self.assertAlmostEqual(
            BOHR_TO_ANGSTROM,
            0.5291772109,
            places=10
        )

        self.assertAlmostEqual(
            HARTREE_TO_EV,
            27.2113862459,
            places=10
        )

        self.assertEqual(HBAR_AU, 1.0)
        self.assertEqual(ELECTRON_MASS_AU, 1.0)

    def test_bohr_to_angstrom(self):
        """Verify conversion from Bohr to Angstrom."""

        self.assertAlmostEqual(
            bohr_to_angstrom(1.0),
            BOHR_TO_ANGSTROM,
            places=10
        )

    def test_angstrom_to_bohr(self):
        """Verify conversion from Angstrom to Bohr."""

        self.assertAlmostEqual(
            angstrom_to_bohr(BOHR_TO_ANGSTROM),
            1.0,
            places=10
        )

    def test_hartree_to_ev(self):
        """Verify conversion from Hartree to eV."""

        self.assertAlmostEqual(
            hartree_to_ev(1.0),
            HARTREE_TO_EV,
            places=10
        )

    def test_ev_to_hartree(self):
        """Verify conversion from eV to Hartree."""

        self.assertAlmostEqual(
            ev_to_hartree(HARTREE_TO_EV),
            1.0,
            places=10
        )

    def test_length_round_trip(self):
        """Verify reversible length conversions."""

        values = [0.0, 0.5, 1.0, 5.0, 10.0]

        for length in values:
            with self.subTest(length=length):

                converted = bohr_to_angstrom(length)
                recovered = angstrom_to_bohr(converted)

                self.assertAlmostEqual(
                    recovered,
                    length,
                    places=12
                )

    def test_energy_round_trip(self):
        """Verify reversible energy conversions."""

        values = [-2.0, -0.5, 0.0, 1.0, 5.0]

        for energy in values:
            with self.subTest(energy=energy):

                converted = hartree_to_ev(energy)
                recovered = ev_to_hartree(converted)

                self.assertAlmostEqual(
                    recovered,
                    energy,
                    places=12
                )

    def test_numpy_length_arrays(self):
        """Verify vectorized length conversions."""

        lengths = np.array([0.0, 1.0, 2.0, 5.0])

        expected = lengths * BOHR_TO_ANGSTROM

        result = bohr_to_angstrom(lengths)

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-12,
            atol=1e-12
        )

    def test_numpy_energy_arrays(self):
        """Verify vectorized energy conversions."""

        energies = np.array([-1.0, 0.0, 0.5, 2.0])

        expected = energies * HARTREE_TO_EV

        result = hartree_to_ev(energies)

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-12,
            atol=1e-12
        )

    def test_general_length_conversion(self):
        """Verify the general length converter."""

        self.assertAlmostEqual(
            convert_length(1.0, "bohr", "angstrom"),
            BOHR_TO_ANGSTROM,
            places=10
        )

        self.assertAlmostEqual(
            convert_length(1.0, "angstrom", "bohr"),
            1.0 / BOHR_TO_ANGSTROM,
            places=10
        )

        self.assertEqual(
            convert_length(5.0, "bohr", "bohr"),
            5.0
        )

    def test_general_energy_conversion(self):
        """Verify the general energy converter."""

        self.assertAlmostEqual(
            convert_energy(1.0, "hartree", "ev"),
            HARTREE_TO_EV,
            places=10
        )

        self.assertAlmostEqual(
            convert_energy(1.0, "ev", "hartree"),
            1.0 / HARTREE_TO_EV,
            places=10
        )

        self.assertEqual(
            convert_energy(5.0, "ev", "ev"),
            5.0
        )

    def test_case_insensitive_units(self):
        """Verify case-insensitive unit names."""

        self.assertAlmostEqual(
            convert_length(1.0, "BOHR", "ANGSTROM"),
            BOHR_TO_ANGSTROM
        )

        self.assertAlmostEqual(
            convert_energy(1.0, "HARTREE", "EV"),
            HARTREE_TO_EV
        )

    def test_invalid_length_units(self):
        """Verify rejection of unsupported length units."""

        with self.assertRaises(ValueError):
            convert_length(1.0, "meters", "bohr")

        with self.assertRaises(ValueError):
            convert_length(1.0, "bohr", "kilometers")

    def test_invalid_energy_units(self):
        """Verify rejection of unsupported energy units."""

        with self.assertRaises(ValueError):
            convert_energy(1.0, "joule", "hartree")

        with self.assertRaises(ValueError):
            convert_energy(1.0, "ev", "calorie")


if __name__ == "__main__":
    unittest.main(verbosity=2)
