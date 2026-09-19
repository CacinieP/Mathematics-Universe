#!/usr/bin/env python3
"""Independent exact/symbolic checks for the worked examples in chapters 01–05.

Run with Python + sympy + scipy. These checks verify mathematical calculations,
not the Markdown renderer; they do not replace review of theorem hypotheses.
"""

import itertools
import math
import unittest
from fractions import Fraction

import sympy as s
from scipy import integrate, stats


x, y, z, t = s.symbols("x y z t", real=True)


class MathCoreExamples(unittest.TestCase):
    def assertSymbolicEqual(self, actual, expected):
        self.assertEqual(s.simplify(actual - expected), 0)

    def test_high_school_sum_identities(self):
        k = s.symbols("k", integer=True, positive=True)
        n = s.symbols("n", integer=True, positive=True)
        self.assertSymbolicEqual(s.summation(1 / (k * (k + 1)), (k, 1, n)), 1 - 1 / (n + 1))
        self.assertSymbolicEqual(s.summation(k**2, (k, 1, n)), n * (n + 1) * (2 * n + 1) / 6)
        for count in range(1, 20):
            self.assertEqual(sum((-1)**j * math.comb(count, j) for j in range(count + 1)), 0)

    def test_flush_and_birthdays(self):
        self.assertEqual(math.comb(13, 5), 1287)
        self.assertEqual(math.comb(52, 5), 2598960)
        self.assertEqual(round(4 * math.comb(13, 5) / math.comb(52, 5), 5), 0.00198)
        self.assertEqual(round(4 * (math.comb(13, 5) - 10) / math.comb(52, 5), 5), 0.00197)
        for n, percent in ((23, 50.7), (50, 97.0), (70, 99.9)):
            p = 1 - math.prod(Fraction(365 - j, 365) for j in range(n))
            self.assertEqual(round(float(p) * 100, 1), percent)

    def test_bayes_examples(self):
        p1 = Fraction(99, 100) * Fraction(1, 1000)
        q1 = Fraction(1, 100) * Fraction(999, 1000)
        self.assertAlmostEqual(float(p1 / (p1 + q1)), 0.0901639344262295)
        p2 = Fraction(95, 100) * Fraction(1, 1000)
        q2 = Fraction(2, 100) * Fraction(999, 1000)
        self.assertEqual(round(float(p2 / (p2 + q2)) * 100, 1), 4.5)
        joint = [Fraction(5, 100) / 6, Fraction(3, 100) * 2 / 6, Fraction(1, 100) * 3 / 6]
        self.assertEqual([p / sum(joint) for p in joint], [Fraction(5, 14), Fraction(3, 7), Fraction(3, 14)])

    def test_monty_hall_conditioning(self):
        # Joint probabilities of car location and observing host open door 3.
        joint = [Fraction(1, 3) * Fraction(1, 2), Fraction(1, 3), Fraction(0)]
        self.assertEqual(sum(joint), Fraction(1, 2))
        self.assertEqual(joint[0] / sum(joint), Fraction(1, 3))
        self.assertEqual(joint[1] / sum(joint), Fraction(2, 3))

    def test_pairwise_but_not_mutual_independence(self):
        outcomes = list(itertools.product((0, 1), repeat=2))
        events = [lambda a: a[0] == 1, lambda a: a[1] == 1, lambda a: a[0] == a[1]]
        probability = lambda es: Fraction(sum(all(e(a) for e in es) for a in outcomes), 4)
        for left, right in itertools.combinations(events, 2):
            self.assertEqual(probability([left, right]), probability([left]) * probability([right]))
        self.assertEqual(probability(events), Fraction(1, 4))
        self.assertNotEqual(probability(events), Fraction(1, 8))

    def test_bridge_limits(self):
        examples = [
            ((x**2 - 1) / (x - 1), 1, 2),
            ((s.sqrt(1 + x) - 1) / x, 0, s.Rational(1, 2)),
            (s.sin(2*x) / (3*x), 0, s.Rational(2, 3)),
            ((s.tan(x) - s.sin(x)) / x**3, 0, s.Rational(1, 2)),
            (((1 + x)**(1/x) - s.E) / x, 0, -s.E/2),
            ((s.exp(x) - 1 - x) / x**2, 0, s.Rational(1, 2)),
            ((s.cos(x) - s.exp(-x**2/2)) / x**4, 0, -s.Rational(1, 12)),
        ]
        for expression, point, answer in examples:
            with self.subTest(expression=expression):
                self.assertSymbolicEqual(s.limit(expression, x, point), answer)
        self.assertSymbolicEqual(s.limit(s.log(s.sin(x)/x) / x**2, x, 0), -s.Rational(1, 6))
        self.assertEqual(s.limit(x**2 / s.exp(x), x, s.oo), 0)

    def test_sequence_limits(self):
        n, k = s.symbols("n k", integer=True, positive=True)
        self.assertSymbolicEqual(s.limit((2*n + 1)/(3*n - 1), n, s.oo), s.Rational(2, 3))
        for power in range(9):
            expression = s.summation(k**power, (k, 1, n)) / n**(power + 1)
            self.assertEqual(s.limit(expression, n, s.oo), s.Rational(1, power + 1))
        self.assertSymbolicEqual(s.summation(1/(2**k * s.factorial(k)), (k, 1, s.oo)), s.exp(s.Rational(1, 2)) - 1)
        value = math.sqrt(2)
        for _ in range(20):
            next_value = math.sqrt(2 + value)
            self.assertLess(value, next_value)
            self.assertLess(next_value, 2)
            value = next_value

    def test_rotation_and_differentiation_matrix(self):
        rotation = s.Matrix([[s.sqrt(3)/2, -s.Rational(1, 2)], [s.Rational(1, 2), s.sqrt(3)/2]])
        self.assertEqual(rotation.T * rotation, s.eye(2))
        self.assertEqual(rotation * s.Matrix([1, 0]), s.Matrix([s.sqrt(3)/2, s.Rational(1, 2)]))
        basis = [1, x, x**2]
        matrix = s.Matrix([[0, 1, 0], [0, 0, 2], [0, 0, 0]])
        for col, polynomial in enumerate(basis):
            self.assertSymbolicEqual(sum(matrix[row, col] * basis[row] for row in range(3)), s.diff(polynomial, x))

    def test_nth_derivative(self):
        for n in range(11):
            expected = 2**s.Integer(n - 2) * s.exp(2*x) * (4*x**2 + 4*n*x + n*(n - 1))
            self.assertSymbolicEqual(s.diff(x**2*s.exp(2*x), x, n), expected)

    def test_antiderivatives(self):
        primitive = s.Rational(4, 5) * s.log(x - 3) + s.Rational(1, 5) * s.log(x + 2)
        self.assertSymbolicEqual(s.diff(primitive, x), (x + 1)/(x**2 - x - 6))
        for a, b in ((2, 3), (0, 3), (3, 0), (-2, 4)):
            primitive = s.exp(a*x) * (a*s.sin(b*x) - b*s.cos(b*x)) / (a*a + b*b)
            self.assertSymbolicEqual(s.diff(primitive, x), s.exp(a*x)*s.sin(b*x))
        self.assertEqual(s.integrate(s.log(x)/x**2, (x, 1, s.oo)), 1)

    def test_wallis_and_gamma_beta(self):
        for n in range(15):
            multiplier = s.pi/2 if n % 2 == 0 else s.Integer(1)
            expected = s.factorial2(n - 1) / s.factorial2(n) * multiplier
            self.assertSymbolicEqual(s.integrate(s.sin(x)**n, (x, 0, s.pi/2)), expected)
        self.assertSymbolicEqual(s.gamma(s.Rational(1, 2)), s.sqrt(s.pi))
        for p, q in ((1, 1), (2, 3), (s.Rational(1, 2), s.Rational(1, 2))):
            actual = s.integrate(x**(p - 1) * (1 - x)**(q - 1), (x, 0, 1))
            self.assertSymbolicEqual(actual, s.gamma(p)*s.gamma(q)/s.gamma(p + q))

    def test_multivariable_derivatives(self):
        f = x*y*(x*x - y*y)/(x*x + y*y)
        # Compute axis partial derivatives by limits, preserving the piecewise value at 0.
        fx_axis = s.limit(f/x, x, 0)
        fy_axis = s.limit(f/y, y, 0)
        self.assertEqual(s.limit(fx_axis/y, y, 0), -1)
        self.assertEqual(s.limit(fy_axis/x, x, 0), 1)
        self.assertSymbolicEqual(s.diff(s.exp(x*x + s.sin(x)**2), x), 2*s.exp(x*x+s.sin(x)**2)*(x+s.sin(x)*s.cos(x)))
        self.assertSymbolicEqual(x*(1-x), s.Rational(1, 4) - (x-s.Rational(1, 2))**2)

    def test_multiple_integrals(self):
        r = s.symbols("r", nonnegative=True)
        radius = s.symbols("R", positive=True)
        self.assertSymbolicEqual(s.integrate(y*s.exp(-y*y), (y, 0, 1)), (1 - s.exp(-1))/2)
        self.assertSymbolicEqual(2*s.pi*s.integrate(r**3, (r, 0, radius)), s.pi*radius**4/2)
        self.assertSymbolicEqual(s.integrate(s.pi*(radius**2-z**2), (z, -radius, radius)), 4*s.pi*radius**3/3)
        self.assertSymbolicEqual(2*s.pi*s.integrate(r*s.exp(-r*r), (r, 0, s.oo)), s.pi)

    def test_closed_one_form_counterexample(self):
        p, q = -y/(x*x+y*y), x/(x*x+y*y)
        self.assertSymbolicEqual(s.diff(q, x), s.diff(p, y))
        integrand = p.subs({x: s.cos(t), y: s.sin(t)}) * -s.sin(t) + q.subs({x: s.cos(t), y: s.sin(t)}) * s.cos(t)
        self.assertEqual(s.integrate(s.simplify(integrand), (t, 0, 2*s.pi)), 2*s.pi)

    def test_series_and_fourier(self):
        n = s.symbols("n", integer=True, positive=True)
        self.assertEqual(s.summation(n/2**n, (n, 1, s.oo)), 2)
        self.assertSymbolicEqual(s.summation(1/n**2, (n, 1, s.oo)), s.pi**2/6)
        for k in range(1, 9):
            self.assertSymbolicEqual(2/s.pi*s.integrate(x*s.sin(k*x), (x, 0, s.pi)), 2*s.Rational((-1)**(k+1), k))
            self.assertSymbolicEqual(2/s.pi*s.integrate(x*s.cos(k*x), (x, 0, s.pi)), 2*((-1)**k-1)/(s.pi*k*k))
        self.assertSymbolicEqual(2/s.pi*s.integrate(x*x, (x, 0, s.pi)), 2*s.pi**2/3)

    def test_nonanalytic_smooth_counterexample(self):
        positive_x = s.symbols("u", positive=True)
        function = s.exp(-1/positive_x**2)
        for n in range(9):
            self.assertEqual(s.limit(s.diff(function, positive_x, n), positive_x, 0, dir="+"), 0)
        self.assertGreater(float(function.subs(positive_x, 1)), 0)

    def test_ode_solutions(self):
        c1, c2 = s.symbols("C1 C2")
        solution = s.exp(-x)*(x + c1)
        self.assertSymbolicEqual(s.diff(solution, x) + solution, s.exp(-x))
        solution = c1*s.exp(x) + c2*s.exp(2*x) - x*s.exp(x)
        self.assertSymbolicEqual(s.diff(solution, x, 2) - 3*s.diff(solution, x) + 2*solution, s.exp(x))
        solution = 20 - 10*s.exp(-t/20)
        self.assertSymbolicEqual(s.diff(solution, t), 1-solution/20)
        self.assertEqual(solution.subs(t, 0), 10)
        self.assertEqual(s.limit(solution, t, s.oo), 20)

    def test_logistic_solution_and_regimes(self):
        rate, capacity, initial = s.symbols("r K N0", positive=True)
        solution = capacity/(1+(capacity/initial-1)*s.exp(-rate*t))
        self.assertSymbolicEqual(s.diff(solution, t), rate*solution*(1-solution/capacity))
        self.assertSymbolicEqual(solution.subs(t, 0), initial)
        self.assertSymbolicEqual(s.limit(solution, t, s.oo), capacity)
        for n0 in (5, 10, 15):
            derivative = s.diff(solution, t).subs({rate: 1, capacity: 10, initial: n0, t: 0})
            self.assertEqual(s.sign(derivative), s.sign(10-n0))

    def test_tridiagonal_determinants(self):
        a, b, c = s.symbols("a b c")
        previous2, previous1 = s.Integer(1), a
        for n in range(2, 7):
            matrix = s.Matrix(n, n, lambda i, j: a if i == j else (b if j == i+1 else (c if i == j+1 else 0)))
            current = s.expand(matrix.det())
            self.assertSymbolicEqual(current, a*previous1-b*c*previous2)
            self.assertEqual(current.subs({a: 2, b: 1, c: 1}), n+1)
            self.assertEqual(current.subs({a: 0, b: 0, c: 3}), 0)
            previous2, previous1 = previous1, current

    def test_zero_and_vandermonde_determinants(self):
        a, b, c, d = s.symbols("a b c d")
        matrix = s.Matrix([[a,b,c,d], [a,b,d,c], [a,c,b,d], [a,c,d,b]])
        self.assertEqual(s.factor(matrix.det()), 0)
        nodes = s.symbols("x1:5")
        matrix = s.Matrix([[v**power for v in nodes] for power in range(4)])
        self.assertSymbolicEqual(matrix.det(), s.prod(nodes[j]-nodes[i] for i in range(4) for j in range(i+1, 4)))

    def test_gram_schmidt(self):
        inputs = [s.Matrix([1,1,0]), s.Matrix([1,0,1]), s.Matrix([0,1,1])]
        orthogonal = s.GramSchmidt(inputs)
        expected = [s.Matrix([1,1,0]), s.Matrix([s.Rational(1,2), -s.Rational(1,2), 1]), s.Matrix([-s.Rational(2,3), s.Rational(2,3), s.Rational(2,3)])]
        self.assertEqual(orthogonal, expected)
        for left, right in itertools.combinations(orthogonal, 2):
            self.assertEqual(left.dot(right), 0)

    def test_linear_systems(self):
        matrix = s.Matrix([[1,1,-1,-1], [2,1,1,1], [1,-1,5,5]])
        self.assertEqual(matrix.rref()[0], s.Matrix([[1,0,2,2], [0,1,-3,-3], [0,0,0,0]]))
        for vector in [s.Matrix([-2,3,1,0]), s.Matrix([-2,3,0,1])]:
            self.assertEqual(matrix*vector, s.zeros(3, 1))
        parameter = s.symbols("lambda")
        matrix = s.ones(3) + parameter*s.eye(3)
        rhs = s.Matrix([0, parameter, parameter**2])
        self.assertSymbolicEqual(matrix.det(), (parameter+3)*parameter**2)
        for value, coefficient_rank, augmented_rank in ((0,1,1), (-3,2,3), (2,3,3)):
            self.assertEqual(matrix.subs(parameter, value).rank(), coefficient_rank)
            self.assertEqual(matrix.row_join(rhs).subs(parameter, value).rank(), augmented_rank)

    def test_eigen_and_matrix_powers(self):
        matrix = s.Matrix([[3,2], [-1,0]])
        for eigenvalue, vector in ((1, s.Matrix([1,-1])), (2, s.Matrix([2,-1]))):
            self.assertEqual(matrix*vector, eigenvalue*vector)
        matrix = s.Matrix([[1,1], [0,2]])
        for n in range(12):
            self.assertEqual(matrix**n, s.Matrix([[1,2**n-1], [0,2**n]]))
        q = s.Matrix([[1,1], [1,-1]])/s.sqrt(2)
        self.assertEqual(q.T*q, s.eye(2))
        self.assertEqual(q*s.diag(1,3)*q.T, s.Matrix([[2,-1], [-1,2]]))
        phi, psi = (1+s.sqrt(5))/2, (1-s.sqrt(5))/2
        for n in range(15):
            self.assertSymbolicEqual((phi**n-psi**n)/s.sqrt(5), s.fibonacci(n))

    def test_quadratic_forms_and_hessian(self):
        self.assertSymbolicEqual(x*x+2*x*y+2*y*y-6*y*z, (x+y)**2+(y-3*z)**2-9*z*z)
        function = x**3+y**3-3*x*y
        self.assertEqual(set(s.solve([s.diff(function,x), s.diff(function,y)], (x,y))), {(0,0), (1,1)})
        hessian = s.hessian(function, (x,y))
        self.assertEqual(hessian.subs({x:1,y:1}).det(), 27)
        self.assertEqual(hessian.subs({x:0,y:0}).det(), -9)
        counterexample = s.diag(0,-1)
        self.assertTrue(all(counterexample[:i,:i].det() >= 0 for i in (1,2)))
        self.assertFalse(counterexample.is_positive_semidefinite)

    def test_density_transform_and_moments(self):
        u = s.symbols("u", positive=True)
        density = s.exp(-u/2)/s.sqrt(2*s.pi*u)
        self.assertEqual(s.integrate(density, (u,0,s.oo)), 1)
        self.assertEqual(s.integrate(u*density, (u,0,s.oo)), 1)
        joint = 2*s.exp(-2*x-y)
        self.assertEqual(s.integrate(joint, (y,0,s.oo)), 2*s.exp(-2*x))
        self.assertEqual(s.integrate(joint, (x,0,s.oo)), s.exp(-y))
        rate = s.symbols("lambda", positive=True)
        convolution = s.integrate(rate*s.exp(-rate*x)*rate*s.exp(-rate*(u-x)), (x,0,u))
        self.assertSymbolicEqual(convolution, rate**2*u*s.exp(-rate*u))
        self.assertEqual(s.integrate(convolution, (u,0,s.oo)), 1)

    def test_uncorrelated_dependence(self):
        ex = s.integrate(x/2, (x,-1,1))
        ey = s.integrate(x*x/2, (x,-1,1))
        exy = s.integrate(x**3/2, (x,-1,1))
        self.assertEqual(exy-ex*ey, 0)
        # X in (-1/2,1/2) iff Y<1/4; intersection probability is 1/2, not 1/4.
        self.assertEqual(s.integrate(s.Rational(1,2), (x,-s.Rational(1,2),s.Rational(1,2))), s.Rational(1,2))
        self.assertEqual(s.integrate(x**4/2, (x,-1,1))-ey**2, s.Rational(4,45))

    def test_fixed_points_expected_count(self):
        for n in range(1, 8):
            total = sum(sum(index == value for index, value in enumerate(permutation)) for permutation in itertools.permutations(range(n)))
            self.assertEqual(Fraction(total, math.factorial(n)), 1)

    def test_normal_probability_and_clt_examples(self):
        self.assertEqual(round(stats.norm.cdf(2)-stats.norm.cdf(-1), 4), 0.8186)
        sd = math.sqrt(3500/12)
        dice_normal = stats.norm.cdf(19.5/sd)-stats.norm.cdf(-19.5/sd)
        self.assertEqual(round(dice_normal, 4), 0.7465)
        # Exact integer convolution: count all 6^100 equiprobable dice outcomes.
        counts = [1]
        for _ in range(100):
            next_counts = [0]*(len(counts)+6)
            for total, count in enumerate(counts):
                for face in range(1,7):
                    next_counts[total+face] += count
            counts = next_counts
        self.assertEqual(sum(counts), 6**100)
        exact_dice = float(Fraction(sum(counts[331:370]), 6**100))
        self.assertLess(abs(exact_dice-dice_normal), 0.001)
        exact_binomial = sum(Fraction(math.comb(100,k)*9**k, 10**100) for k in range(85,101))
        self.assertAlmostEqual(float(exact_binomial), stats.binom.sf(84,100,0.9), places=14)
        self.assertEqual(round(float(exact_binomial), 4), 0.9601)
        self.assertEqual(round(stats.norm.cdf(11/6), 4), 0.9666)
        self.assertEqual(round(stats.norm.cdf(5/3), 4), 0.9522)
        print(f"\nVerified probabilities: dice exact={exact_dice:.10f}, normal={dice_normal:.10f}; binomial exact={float(exact_binomial):.10f}")

    def test_buffon_and_conditional_normal(self):
        # Integrate probability of intersection conditional on uniform angle.
        needle, spacing = 0.7, 1.3
        probability, _ = integrate.quad(lambda angle: needle/spacing*math.sin(angle), 0, math.pi)
        self.assertAlmostEqual(probability/math.pi, 2*needle/(math.pi*spacing))
        rho, sx, sy = s.symbols("rho sx sy", real=True)
        sigma = s.Matrix([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        self.assertSymbolicEqual(sigma.det(), sx*sx*sy*sy*(1-rho*rho))
        self.assertSymbolicEqual(sigma[0,0] - sigma[0,1]**2 / sigma[1,1], sx*sx*(1-rho*rho))

    def test_variance_estimator_identity_and_normal_mle(self):
        observations = [s.Integer(1), s.Integer(2), s.Integer(4), s.Integer(5)]
        mean = sum(observations)/len(observations)
        ss = sum((value-mean)**2 for value in observations)
        self.assertEqual(ss, sum(value**2 for value in observations)-len(observations)*mean**2)
        mu = s.symbols("mu", real=True)
        variance = s.symbols("v", positive=True)
        likelihood = -len(observations)/2*s.log(variance)-sum((value-mu)**2 for value in observations)/(2*variance)
        # Differentiate treating variance itself as the parameter.
        solution = {mu: mean, variance: ss/len(observations)}
        self.assertEqual(s.diff(likelihood,mu).subs(solution), 0)
        self.assertEqual(s.diff(likelihood,variance).subs(solution), 0)
        self.assertTrue(s.hessian(likelihood,(mu,variance)).subs(solution).is_negative_definite)
        flat_likelihood = -len(observations)*s.log(variance)/2
        self.assertEqual(s.limit(flat_likelihood, variance, 0, dir="+"), s.oo)

    def test_upper_tail_quantiles_and_battery_test(self):
        self.assertEqual(round(stats.norm.isf(0.05), 3), 1.645)
        self.assertEqual(round(stats.norm.isf(0.025), 2), 1.96)
        for df in (1, 5, 24):
            lower_pivot = stats.chi2.isf(0.975,df)
            upper_pivot = stats.chi2.isf(0.025,df)
            self.assertLess(lower_pivot, upper_pivot)
            self.assertAlmostEqual(stats.chi2.cdf(upper_pivot,df)-stats.chi2.cdf(lower_pivot,df), 0.95)
            self.assertLess(df/upper_pivot, df/lower_pivot)
        statistic = (488-500)/(50/math.sqrt(25))
        self.assertEqual(statistic, -1.2)
        self.assertGreater(stats.norm.cdf(statistic), 0.05)


if __name__ == "__main__":
    unittest.main(verbosity=2)
