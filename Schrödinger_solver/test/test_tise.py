
"""
Numerical tests for the Schrodinger1D TISE solver.

Tests:
    - Finite-difference matrices
    - Hamiltonian symmetry
    - Infinite square well eigenenergies
    - Wavefunction normalization
    - Eigenstate orthogonality
    - Periodic boundary conditions
    - Numerical convergence
    - Constant potential energy shift
    - Eigenvalue equation residuals
    - Invalid input parameters
"""

import unittest
import numpy as np

from packets.schrodinger1d.tise import (
    second_derivative_matrix,
    build_hamiltonian,
    solve_tise
)


class TestTISE(unittest.TestCase):
    """Tests for the time-independent Schrodinger solver."""

    def setUp(self):
        """Define common parameters for numerical tests."""

        self.L = 10.0
        self.N = 200
        self.h = self.L / (self.N + 1)
        self.mass = 1.0
        self.potential = np.zeros(self.N)

    def test_dirichlet_matrix(self):
        """Verify the second-derivative matrix."""

        n = 4
        h = 0.5

        D2 = second_derivative_matrix(
            n=n,
            h=h,
            boundary="dirichlet"
        ).toarray()

        expected = np.array([
            [-2,  1,  0,  0],
            [ 1, -2,  1,  0],
            [ 0,  1, -2,  1],
            [ 0,  0,  1, -2]
        ]) / h**2

        np.testing.assert_allclose(
            D2,
            expected,
            atol=1e-12
        )

    def test_periodic_matrix(self):
        """Verify periodic connections between endpoints."""

        n = 5
        h = 0.5

        D2 = second_derivative_matrix(
            n=n,
            h=h,
            boundary="periodic"
        ).toarray()

        expected = np.array([
            [-2,  1,  0,  0,  1],
            [ 1, -2,  1,  0,  0],
            [ 0,  1, -2,  1,  0],
            [ 0,  0,  1, -2,  1],
            [ 1,  0,  0,  1, -2]
        ]) / h**2

        np.testing.assert_allclose(
            D2,
            expected,
            atol=1e-12
        )

    def test_hamiltonian_symmetry(self):
        """Verify that the Hamiltonian is symmetric."""

        x = np.arange(1, self.N + 1) * self.h
        potential = 0.02 * (x - self.L / 2)**2

        H = build_hamiltonian(
            potential=potential,
            h=self.h,
            boundary="dirichlet",
            mass=self.mass
        ).toarray()

        np.testing.assert_allclose(
            H,
            H.T,
            atol=1e-12
        )

    def test_infinite_square_well_energies(self):
        """Compare numerical and analytical energies."""

        result = solve_tise(
            potential=self.potential,
            h=self.h,
            L=self.L,
            boundary="dirichlet",
            mass=self.mass,
            num_states=5
        )

        numerical = result["energies"]

        n = np.arange(1, 6)

        analytical = (
            n**2 * np.pi**2
            / (2.0 * self.mass * self.L**2)
        )

        np.testing.assert_allclose(
            numerical,
            analytical,
            rtol=1e-3,
            atol=0
        )

    def test_wavefunction_normalization(self):
        """Verify spatial normalization of eigenfunctions."""

        result = solve_tise(
            potential=self.potential,
            h=self.h,
            L=self.L,
            num_states=5
        )

        psi = result["wavefunctions"]

        normalization = (
            self.h * np.sum(np.abs(psi)**2, axis=0)
        )

        np.testing.assert_allclose(
            normalization,
            np.ones(5),
            atol=1e-10
        )

    def test_eigenstate_orthogonality(self):
        """Verify orthonormality of eigenfunctions."""

        result = solve_tise(
            potential=self.potential,
            h=self.h,
            L=self.L,
            num_states=5
        )

        psi = result["wavefunctions"]

        overlap = self.h * psi.conj().T @ psi

        np.testing.assert_allclose(
            overlap,
            np.eye(5),
            atol=1e-10
        )

    def test_periodic_free_particle(self):
        """Verify periodic free-particle energy levels."""

        n = 120
        L = 10.0
        h = L / n

        potential = np.zeros(n)

        result = solve_tise(
            potential=potential,
            h=h,
            L=L,
            boundary="periodic",
            num_states=3
        )

        energies = result["energies"]

        # Ground-state energy.
        self.assertAlmostEqual(
            energies[0],
            0.0,
            delta=1e-9
        )

        # Exact first excited energy of the discrete matrix.
        expected = (
            2.0 * np.sin(np.pi / n)**2 / h**2
        )

        np.testing.assert_allclose(
            energies[1:3],
            [expected, expected],
            rtol=1e-8,
            atol=1e-9
        )

    def test_grid_convergence(self):
        """Verify decreasing error for a finer grid."""

        L = 10.0
        analytical = np.pi**2 / (2.0 * L**2)

        # Coarse grid.
        n_coarse = 40
        h_coarse = L / (n_coarse + 1)

        coarse = solve_tise(
            potential=np.zeros(n_coarse),
            h=h_coarse,
            L=L,
            num_states=1
        )

        # Fine grid.
        n_fine = 80
        h_fine = L / (n_fine + 1)

        fine = solve_tise(
            potential=np.zeros(n_fine),
            h=h_fine,
            L=L,
            num_states=1
        )

        error_coarse = abs(
            coarse["energies"][0] - analytical
        )

        error_fine = abs(
            fine["energies"][0] - analytical
        )

        self.assertLess(error_fine, error_coarse)

    def test_constant_potential_shift(self):
        """Verify the effect of adding a constant potential."""

        V0 = 0.3

        free = solve_tise(
            potential=np.zeros(self.N),
            h=self.h,
            L=self.L,
            num_states=4
        )

        shifted = solve_tise(
            potential=np.full(self.N, V0),
            h=self.h,
            L=self.L,
            num_states=4
        )

        np.testing.assert_allclose(
            shifted["energies"],
            free["energies"] + V0,
            atol=1e-10
        )

    def test_eigenvalue_equation(self):
        """Verify H psi = E psi numerically."""

        result = solve_tise(
            potential=self.potential,
            h=self.h,
            L=self.L,
            num_states=4
        )

        H = build_hamiltonian(
            potential=self.potential,
            h=self.h
        )

        psi = result["wavefunctions"]
        energies = result["energies"]

        residual = (
            H @ psi - psi * energies[np.newaxis, :]
        )

        self.assertLess(
            np.max(np.abs(residual)),
            1e-9
        )

    def test_invalid_parameters(self):
        """Verify rejection of invalid numerical parameters."""

        with self.assertRaises(ValueError):
            second_derivative_matrix(
                n=10,
                h=0.0
            )

        with self.assertRaises(ValueError):
            build_hamiltonian(
                potential=np.zeros(10),
                h=0.1,
                mass=-1.0
            )

        with self.assertRaises(ValueError):
            solve_tise(
                potential=np.zeros(10),
                h=0.1,
                boundary="invalid"
            )

        with self.assertRaises(ValueError):
            solve_tise(
                potential=np.zeros(10),
                h=0.1,
                num_states=20
            )

        with self.assertRaises(ValueError):
            solve_tise(
                potential=np.zeros(10),
                h=0.1,
                L=100.0,
                num_states=2
            )



    def test_second_order_convergence(self):
        """Verify second-order convergence of finite differences."""

        L = 10.0
        analytical = np.pi**2 / (2.0 * L**2)

        errors = []

        for n in [40, 80, 160]:
            h = L / (n + 1)

            result = solve_tise(
                potential=np.zeros(n),
                h=h,
                L=L,
                num_states=1
            )

            error = abs(result["energies"][0] - analytical)
            errors.append(error)

        p1 = (
            np.log(errors[0] / errors[1])
            / np.log(81 / 41)
        )

        p2 = (
            np.log(errors[1] / errors[2])
            / np.log(161 / 81)
        )

        self.assertAlmostEqual(p1, 2.0, delta=0.1)
        self.assertAlmostEqual(p2, 2.0, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
