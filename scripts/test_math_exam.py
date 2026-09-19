"""Independent mathematical recalculations for the 2026-09 audit.

Run with Python 3 + SymPy: python scripts/test_math_exam.py
These are algebra/integration checks, not a Markdown parser or formal proofs.
General existence and convergence arguments were separately reviewed by hand.
"""

import unittest
from math import comb

import sympy as s

x, y, t = s.symbols('x y t', real=True)
a, b = s.symbols('a b', real=True)
theta = s.symbols('theta', positive=True)
R = s.Rational


class ExamCalculations(unittest.TestCase):
    def eq(self, actual, expected):
        self.assertEqual(s.simplify(actual - expected), 0)

    def test_c01_riemann_integral(self):
        self.eq(s.integrate(x / (1 + x*x), (x, 0, 1)), s.log(2)/2)

    def test_c02_jump_limits(self):
        f = x / (1 + s.exp(1/(1-x)))
        self.eq(s.limit(f, x, 1, dir='-'), 0)
        self.eq(s.limit(f, x, 1, dir='+'), 1)

    def test_c03_curvature(self):
        f = x*x/2 - s.log(1+x*x)/4
        slope, bend = s.diff(f, x).subs(x, 1), s.diff(f, x, 2).subs(x, 1)
        self.eq(abs(bend) / (1+slope*slope)**R(3, 2), R(64, 125))

    def test_c05_series_radius_and_sum(self):
        n = s.symbols('n', integer=True, positive=True)
        self.eq(s.limit((n+1)*2/n, n, s.oo), 2)
        self.eq(s.diff(-s.log(1-x/2), x), 1/(2-x))
        self.eq((-s.log(1-x/2)).subs(x, -2), -s.log(2))

    def test_c07_gaussian_bounds(self):
        self.eq(s.integrate(s.exp(-x*x), (x, 0, s.oo)), s.sqrt(s.pi)/2)

    def test_c08_constrained_extrema(self):
        A = s.Matrix([[1, R(1, 2)], [R(1, 2), 1]])
        self.assertEqual(A.eigenvals(), {R(1, 2): 1, R(3, 2): 1})

    def test_c09_derivative_coefficient_identity(self):
        # General proof also needs the two difference quotients -> f'(0).
        self.eq(1/(1-x) - x/(1-x), 1)
        self.eq(s.limit(x/(1-x), x, 0), 0)

    def test_c11_improper_integrals(self):
        self.assertEqual(s.integrate(s.log(x)/x, (x, 1, s.oo)), s.oo)
        self.assertEqual(s.integrate(x/(1+x*x), (x, 0, s.oo)), s.oo)
        self.assertEqual(s.limit(s.log(-s.log(x)), x, 1, dir='-'), -s.oo)
        self.eq(s.limit(1/(s.sqrt(x)*s.log(x)), x, s.oo), 0)

    def test_c12_split_double_integral(self):
        # Integrate the two pieces independently before combining.
        I1 = 2*s.integrate(s.integrate(s.sqrt(x*x-y), (y, 0, x*x)), (x, 0, 1))
        I2 = R(16, 3)*s.integrate(s.cos(t)**4, (t, 0, s.pi/4))
        self.eq(I1, R(1, 3))
        self.eq(I1+I2, s.pi/2+R(5, 3))

    def test_c15_double_integral(self):
        self.eq(s.integrate(x*y, (y, x*x, x), (x, 0, 1)), R(1, 24))

    def test_c16_exponential_generating_function(self):
        self.eq(x*s.diff(x*s.diff(s.exp(x), x), x)+s.exp(x), s.exp(x)*(x*x+x+1))

    def test_c18_all_stationary_points_and_hessians(self):
        f = (x*x-y*y)*s.exp(-x*x-y*y)
        points = s.solve([x*(1-x*x+y*y), y*(1+x*x-y*y)], (x, y))
        self.assertEqual(set(points), {(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)})
        H = s.hessian(f, (x, y))
        self.assertEqual(H.subs({x: 0, y: 0}).det(), -4)
        for xx in (-1, 1):
            eigen = H.subs({x: xx, y: 0}).eigenvals()
            self.assertTrue(all(v < 0 for v in eigen))
            self.eq(f.subs({x: xx, y: 0}), s.exp(-1))
        for yy in (-1, 1):
            eigen = H.subs({x: 0, y: yy}).eigenvals()
            self.assertTrue(all(v > 0 for v in eigen))
            self.eq(f.subs({x: 0, y: yy}), -s.exp(-1))

    def test_l01_adjugate(self):
        A = s.diag(-2, 1, -3)
        self.eq((A.adjugate()+3*s.eye(3)).det(), 0)

    def test_l03_and_l09_orthogonal_diagonalization(self):
        Q = s.Matrix.hstack(s.Matrix([1,1,1])/s.sqrt(3),
                            s.Matrix([1,-1,0])/s.sqrt(2),
                            s.Matrix([1,1,-2])/s.sqrt(6))
        self.assertEqual(s.simplify(Q.T*Q), s.eye(3))
        for A, D in [(s.ones(3), s.diag(3,0,0)), (3*s.eye(3)-s.ones(3), s.diag(0,3,3))]:
            self.assertEqual(s.simplify(Q.T*A*Q), D)

    def test_l04_matrix_counterexamples(self):
        A = s.diag(1,0)
        B = s.Matrix([[0,1],[0,0]])
        self.assertEqual(A*A, A)
        self.assertNotEqual(A, s.eye(2))
        self.assertNotEqual(A, s.zeros(2))
        self.assertEqual(A*s.diag(0,1), s.zeros(2))
        self.assertNotEqual(A*B, B*A)

    def test_l05_parameters_and_spectrum(self):
        sol = s.solve([a-1-(a+b), b-1-(a+b)], (a,b))
        self.assertEqual(sol, {a: -1, b: -1})
        self.assertEqual((s.eye(3)-s.ones(3)).eigenvals(), {-2: 1, 1: 2})

    def test_l06_rank_cases(self):
        A = s.Matrix([[1,a,1],[a,1,b],[1,b,1]])
        self.eq(A.det(), -(a-b)**2)
        self.eq(A[:2,:2].det(), 1-a*a)
        for aa in (-1, 1):
            self.assertEqual(A.subs({a:aa,b:aa}).rank(), 1)

    def test_l07_projection_spectrum(self):
        A = s.diag(1,1,0)
        self.assertEqual(A*A, A)
        self.eq((A-2*s.eye(3)).det(), -2)

    def test_l10_adjugate_transpose_nondiagonalizable(self):
        A = s.Matrix([[2,1,0],[0,2,0],[0,0,3]])
        self.assertEqual(A.adjugate().T.eigenvals(), {6:2,4:1})

    def test_p01_and_p02_basic_probabilities(self):
        self.eq(s.integrate(s.exp(-x*x/2)/s.sqrt(2*s.pi), (x, -s.oo, 0)), R(1,2))
        self.eq(sum(R(1,4) for xx in (0,1) for yy in (0,1) if xx == yy), R(1,2))

    def test_p03_gamma_moments_and_cdf(self):
        density = x*s.exp(-x)
        self.eq(s.integrate(density, (x,0,s.oo)), 1)
        mean = s.integrate(x*density, (x,0,s.oo))
        self.eq(mean, 2)
        self.eq(s.integrate(x*x*density, (x,0,s.oo))-mean**2, 2)
        self.eq(s.diff(1-(x+1)*s.exp(-x), x), density)

    def test_p04_joint_probability(self):
        self.eq(s.integrate(2*s.exp(-2*x-y), (x,y,s.oo), (y,0,s.oo)), R(1,3))

    def test_p05_moments_mle_and_bias(self):
        density = 2*x/theta*s.exp(-x*x/theta)
        mean = s.integrate(x*density, (x,0,s.oo))
        second = s.integrate(x*x*density, (x,0,s.oo))
        self.eq(mean, s.sqrt(s.pi*theta)/2)
        self.eq(second, theta)
        n, T = s.symbols('n T', positive=True)
        self.eq(4/s.pi*((second-mean**2)/n+mean**2), theta*(1+(4-s.pi)/(s.pi*n)))
        self.eq(s.diff(-n*s.log(theta)-T/theta, theta), (T-n*theta)/theta**2)

    def test_p06_conditional_density(self):
        # Use u=1-y>0 to avoid CAS logarithm branch ambiguity at the endpoint.
        u = s.symbols('u', positive=True)
        self.eq(s.integrate(-s.log(u), (u,0,1)), 1)
        self.eq(s.integrate(-(1-u)*s.log(u), (u,0,1)), R(3,4))
        self.eq(s.integrate((x+1)/2, (x,0,1)), R(3,4))

    def test_p08_hypergeometric_all_small_parameters(self):
        for N in range(2, 11):
            for M in range(N+1):
                for n in range(1,N+1):
                    pmf = {k: R(comb(M,k)*comb(N-M,n-k), comb(N,n))
                           for k in range(max(0,n+M-N), min(n,M)+1)}
                    self.eq(sum(pmf.values()), 1)
                    mean = sum(k*v for k,v in pmf.items())
                    self.eq(mean, R(n*M,N))
                    self.eq(sum((k-mean)**2*v for k,v in pmf.items()),
                            R(n*M*(N-M)*(N-n), N*N*(N-1)))

    def test_p09_bernoulli_estimator_moments(self):
        p = s.symbols('p')
        for n in range(1, 9):
            pmf = [s.binomial(n,k)*p**k*(1-p)**(n-k) for k in range(n+1)]
            self.eq(sum(R(k,n)*pmf[k] for k in range(n+1)), p)
            self.eq(sum((R(k,n)-p)**2*pmf[k] for k in range(n+1)), p*(1-p)/n)

    def test_extra_quadratic_form_and_ode(self):
        A = s.Matrix([[1,a,1],[a,1,1],[1,1,1]])
        self.eq(A.det(), -(a-1)**2)
        B = s.Matrix([[2,1],[-1,4]])
        N = B-3*s.eye(2)
        self.assertEqual(N*N, s.zeros(2))
        Y = s.exp(3*t)*(s.eye(2)+t*N)
        self.assertEqual(s.simplify(s.diff(Y,t)-B*Y), s.zeros(2))

    def test_extra_resonance_and_zero_factor(self):
        # Root lambda=3 has multiplicity 2, so the trial multiplier is x**2.
        particular = x*x*s.exp(3*x)/2
        self.eq(s.diff(particular,x,2)-6*s.diff(particular,x)+9*particular, s.exp(3*x))
        # F=x**2*f, f=1: F'(0)=0 does not imply x*f'+2*f=0 at zero.
        self.eq(s.diff(x*x,x).subs(x,0), 0)
        self.assertNotEqual((x*s.diff(s.Integer(1),x)+2).subs(x,0), 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
