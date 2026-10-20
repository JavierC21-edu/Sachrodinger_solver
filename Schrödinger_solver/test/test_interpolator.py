
"""
Unit tests for potential interpolation in Schrodinger1D.

Tests:
    - Dirichlet and periodic grids
    - Grid spacing and domain length
    - Linear interpolation
    - Cubic spline interpolation
    - PCHIP interpolation
    - Periodic potential interpolation
    - Potential data validation
    - Invalid parameter handling
    - Compatibility with the TISE solver
"""

import unittest
import numpy as np

from packets.schrodinger1d.interpolator import (
    create_grid,
    validate_potential_data,
    interpolate_potential,
    interpolate_for_tise
)

from packets.schrodinger1d.tise import solve_tise


class TestInterpolator(unittest.TestCase):
    """Tests for spatial grids and potential interpolation."""

    def setUp(self):
        """Define common numerical parameters."""

        self.L = 10.0
        self.h = 0.05

        self.x = np.linspace(0.0, self.L, 11)

        self.potential = (
            0.02 * (self.x - self.L / 2.0)**2
        )

    def test_dirichlet_grid(self):
        """Verify the non-periodic spatial grid."""

        grid = create_grid(
            L=10.0,
            h=1.0,
            boundary="dirichlet"
        )

        expected = np.arange(1.0, 10.0)

        np.testing.assert_allclose(
            grid["x"],
            expected
        )

        self.assertEqual(grid["h"], 1.0)
        self.assertEqual(grid["L"], 10.0)

    def test_periodic_grid(self):
        """Verify the periodic spatial grid."""

        grid = create_grid(
            L=10.0,
            h=1.0,
            boundary="periodic"
        )

        expected = np.arange(0.0, 10.0)

        np.testing.assert_allclose(
            grid["x"],
            expected
        )

        self.assertEqual(len(grid["x"]), 10)

    def test_grid_spacing(self):
        """Verify uniform spacing between grid points."""

        grid = create_grid(
            L=self.L,
            h=self.h,
            boundary="dirichlet"
        )

        differences = np.diff(grid["x"])

        np.testing.assert_allclose(
            differences,
            self.h,
            rtol=1e-12,
            atol=1e-12
        )

    def test_adjust_grid_spacing(self):
        """Verify automatic grid spacing adjustment."""

        grid = create_grid(
            L=10.0,
            h=0.3,
            boundary="periodic",
            adjust_spacing=True
        )

        expected_intervals = round(10.0 / 0.3)
        expected_h = 10.0 / expected_intervals

        self.assertAlmostEqual(
            grid["h"],
            expected_h,
            places=12
        )

        self.assertEqual(
            len(grid["x"]),
            expected_intervals
        )

    def test_invalid_grid_spacing(self):
        """Verify rejection of incompatible spacing."""

        with self.assertRaises(ValueError):
            create_grid(
                L=10.0,
                h=0.3,
                adjust_spacing=False
            )

    def test_linear_interpolation(self):
        """Verify exact reproduction of a linear function."""

        x = np.array([0.0, 1.0, 2.0, 3.0])
        V = 2.0 * x + 1.0

        x_new = np.linspace(0.0, 3.0, 31)

        result = interpolate_potential(
            x=x,
            potential=V,
            x_new=x_new,
            method="linear"
        )

        expected = 2.0 * x_new + 1.0

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-12,
            atol=1e-12
        )

    def test_cubic_interpolation(self):
        """Verify cubic spline interpolation."""

        x_new = np.linspace(0.0, self.L, 101)

        result = interpolate_potential(
            x=self.x,
            potential=self.potential,
            x_new=x_new,
            method="cubic"
        )

        expected = (
            0.02 * (x_new - self.L / 2.0)**2
        )

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=1e-12
        )

    def test_pchip_interpolation(self):
        """Verify PCHIP interpolation at original points."""

        result = interpolate_potential(
            x=self.x,
            potential=self.potential,
            x_new=self.x,
            method="pchip"
        )

        np.testing.assert_allclose(
            result,
            self.potential,
            rtol=1e-12,
            atol=1e-12
        )

    def test_pchip_monotonicity(self):
        """Verify shape preservation for monotonic data."""

        x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        V = np.array([0.0, 0.5, 1.0, 2.0, 5.0])

        x_new = np.linspace(0.0, 4.0, 101)

        result = interpolate_potential(
            x=x,
            potential=V,
            x_new=x_new,
            method="pchip"
        )

        self.assertTrue(
            np.all(np.diff(result) >= -1e-12)
        )

    def test_periodic_cubic_interpolation(self):
        """Verify interpolation of a periodic potential."""

        L = 10.0

        x = np.linspace(0.0, L, 21)

        V = np.cos(2.0 * np.pi * x / L)

        x_new = np.linspace(0.0, L, 201)

        result = interpolate_potential(
            x=x,
            potential=V,
            x_new=x_new,
            method="cubic",
            periodic=True
        )

        expected = np.cos(
            2.0 * np.pi * x_new / L
        )

        np.testing.assert_allclose(
            result,
            expected,
            rtol=0,
            atol=1e-4
        )

        self.assertAlmostEqual(
            result[0],
            result[-1],
            places=10
        )

    def test_periodic_endpoint_reconstruction(self):
        """Verify periodic input without duplicate endpoint."""

        L = 10.0

        x = np.linspace(
            0.0,
            L,
            20,
            endpoint=False
        )

        V = np.cos(2.0 * np.pi * x / L)

        data = interpolate_for_tise(
            x=x,
            potential=V,
            L=L,
            h=0.05,
            boundary="periodic",
            method="cubic"
        )

        self.assertEqual(
            len(data["x"]),
            200
        )

        self.assertAlmostEqual(
            data["potential"][0],
            V[0],
            places=10
        )

        self.assertTrue(
            np.all(np.isfinite(data["potential"]))
        )

    def test_dirichlet_interpolation_for_tise(self):
        """Verify interpolation onto an interior grid."""

        data = interpolate_for_tise(
            x=self.x,
            potential=self.potential,
            L=self.L,
            h=self.h,
            boundary="dirichlet",
            method="cubic"
        )

        self.assertEqual(
            len(data["x"]),
            199
        )

        self.assertEqual(
            len(data["potential"]),
            199
        )

        expected = (
            0.02 * (data["x"] - self.L / 2.0)**2
        )

        np.testing.assert_allclose(
            data["potential"],
            expected,
            atol=1e-10
        )

    def test_original_data_unchanged(self):
        """Verify that interpolation does not modify input data."""

        x_original = self.x.copy()
        V_original = self.potential.copy()

        interpolate_for_tise(
            x=self.x,
            potential=self.potential,
            L=self.L,
            h=self.h,
            boundary="dirichlet",
            method="linear"
        )

        np.testing.assert_array_equal(
            self.x,
            x_original
        )

        np.testing.assert_array_equal(
            self.potential,
            V_original
        )

    def test_invalid_potential_data(self):
        """Verify validation of malformed input arrays."""

        with self.assertRaises(ValueError):
            validate_potential_data(
                [0, 1, 2],
                [0, 1]
            )

        with self.assertRaises(ValueError):
            validate_potential_data(
                [0, 2, 1],
                [0, 1, 2]
            )

        with self.assertRaises(ValueError):
            validate_potential_data(
                [0, 1, 2],
                [0, np.nan, 2]
            )

    def test_invalid_interpolation_method(self):
        """Verify rejection of unsupported methods."""

        with self.assertRaises(ValueError):
            interpolate_potential(
                x=self.x,
                potential=self.potential,
                x_new=self.x,
                method="invalid"
            )

    def test_extrapolation_rejected(self):
        """Verify that extrapolation is not permitted."""

        x_new = np.array([-1.0, 0.0, 5.0])

        with self.assertRaises(ValueError):
            interpolate_potential(
                x=self.x,
                potential=self.potential,
                x_new=x_new,
                method="linear"
            )

    def test_inconsistent_periodic_potential(self):
        """Verify rejection of mismatched periodic endpoints."""

        x = np.linspace(0.0, 10.0, 11)
        V = np.linspace(0.0, 1.0, 11)

        with self.assertRaises(ValueError):
            interpolate_for_tise(
                x=x,
                potential=V,
                L=10.0,
                h=0.05,
                boundary="periodic",
                method="cubic"
            )

    def test_invalid_domain(self):
        """Verify rejection of invalid domain parameters."""

        with self.assertRaises(ValueError):
            create_grid(
                L=-10.0,
                h=0.05
            )

        with self.assertRaises(ValueError):
            create_grid(
                L=10.0,
                h=-0.05
            )

        with self.assertRaises(ValueError):
            create_grid(
                L=10.0,
                h=0.05,
                boundary="invalid"
            )

    def test_tise_integration(self):
        """Verify compatibility with the Schrodinger solver."""

        # Interpolate a harmonic potential.
        data = interpolate_for_tise(
            x=self.x,
            potential=self.potential,
            L=self.L,
            h=self.h,
            boundary="dirichlet",
            method="cubic"
        )

        # Solve the time-independent Schrodinger equation.
        solution = solve_tise(
            potential=data["potential"],
            h=data["h"],
            L=data["L"],
            boundary=data["boundary"],
            mass=1.0,
            num_states=5
        )

        energies = solution["energies"]
        psi = solution["wavefunctions"]

        # Verify physical and numerical properties.
        self.assertEqual(
            len(energies),
            5
        )

        self.assertEqual(
            psi.shape,
            (199, 5)
        )

        self.assertTrue(
            np.all(np.diff(energies) >= -1e-10)
        )

        self.assertTrue(
            np.all(np.isfinite(energies))
        )

        normalization = (
            data["h"] * np.sum(np.abs(psi)**2, axis=0)
        )

        np.testing.assert_allclose(
            normalization,
            np.ones(5),
            atol=1e-10
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
