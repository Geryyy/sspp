import unittest
import numpy as np
from sspp.BSplines import B, dB, bspline, bspline_derivative, knot_vector, compute_control_points

class TestBSplines(unittest.TestCase):
    
    def setUp(self):
        """Initialize test parameters."""
        self.k = 3  # Cubic B-spline
        self.n_control_points = 5
        self.t = knot_vector(self.n_control_points, self.k)
        self.c = np.array([0, 1, 2, 3, 4])  # Simple control points
    
    # def test_basis_function_sum(self):
    #     """Test that B-spline basis functions sum to 1."""
    #     theta_vals = np.linspace(0, 1, 100)
    #     basis_sums = [sum(B(theta, self.k, i, self.t) for i in range(self.n_control_points)) for theta in theta_vals]
    #     print("basis sum: ", basis_sums)
    #     self.assertTrue(np.allclose(basis_sums, 1, atol=1e-6))

    def test_B_spline_basis(self):
        """Test the B-spline basis function properties."""
        theta = 0.5
        i = 2
        result = B(theta, self.k, i, self.t)
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_B_spline_basis_derivative(self):
        """Test the derivative of the B-spline basis function."""
        theta = 0.5
        i = 2
        result = dB(theta, self.k, i, self.t)
        self.assertIsInstance(result, float)
        
    def test_bspline_evaluation(self):
        """Test the B-spline curve evaluation at a given theta."""
        theta = 0.5
        result = bspline(theta, self.t, self.c, self.k)
        self.assertIsInstance(result, float)
        
    def test_bspline_derivative(self):
        """Test the derivative of the B-spline curve."""
        theta = 0.5
        result = bspline_derivative(theta, self.t, self.c, self.k)
        self.assertIsInstance(result, float)
    
    def test_knot_vector_generation(self):
        """Test knot vector generation."""
        t = knot_vector(self.n_control_points, self.k)
        self.assertEqual(len(t), self.n_control_points + self.k + 1)
        self.assertTrue(np.all(t[:self.k] == 0))  # Ensure k repeated knots at start
        self.assertTrue(np.all(t[-self.k:] == 1))  # Ensure k repeated knots at end
    
    def test_compute_control_points(self):
        """Test computation of control points from via points."""
        via_points = np.array([[0, 0], [1, 2], [2, 3], [3, 5], [4, 6]])
        control_points, t = compute_control_points(via_points, self.k)
        self.assertEqual(control_points.shape, via_points.shape)
        self.assertEqual(len(t), len(self.t))
    
    def test_constant_bspline(self):
        """If all control points are the same, the spline should return the same value."""
        c_const = np.ones((7, 9))  # 7 via points, 9D DoF
        t_const = knot_vector(len(c_const), self.k)
        theta_vals = np.linspace(0, 1, 100)
        y_vals = np.array([bspline(theta, t_const, c_const, self.k) for theta in theta_vals])
        
        for dim in range(9):
            self.assertTrue(np.allclose(y_vals[:, dim], 1), f"Dimension {dim} is not constant!")
    
    def test_constant_bspline(self):
        """Test a trivial case where via points are linearly increasing."""
        k = 3
        c_linear = np.ones((7,1))  # 7 via points, 1D linear increase
        t_linear = knot_vector(len(c_linear), k)
        theta_vals = np.linspace(0, 1, 100)
        y_vals = np.array([bspline(theta, t_linear, c_linear, k) for theta in theta_vals])
        expected_vals = np.linspace(1, 1, 100).reshape(100, 1)

        self.assertTrue(np.allclose(y_vals, expected_vals, atol=1e-6))


    def test_linear_bspline(self):
        """Test a trivial case where via points are linearly increasing."""
        k = 1
        c_linear = np.arange(7).reshape(7, 1)  # 7 via points, 1D linear increase
        t_linear = knot_vector(len(c_linear), k)
        theta_vals = np.linspace(0, 1, 100)
        y_vals = np.array([bspline(theta, t_linear, c_linear, k) for theta in theta_vals])
        expected_vals = np.linspace(0, 6, 100).reshape(100, 1)

        self.assertTrue(np.allclose(y_vals, expected_vals, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
