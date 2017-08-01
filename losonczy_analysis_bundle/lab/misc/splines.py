import numpy as np
from scipy.misc import factorial
knots = np.linspace(-np.pi, np.pi, 100)
K = len(knots)

import numpy as np
from numpy.linalg import svd
from scipy.special import i0
import scipy.optimize


def rank(A, atol=1e-13, rtol=0):
    """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length n will be treated
        as a 2-D with shape (1, n)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    r : int
        The estimated rank of the matrix.

    See also
    --------
    numpy.linalg.matrix_rank
        matrix_rank is basically the same as this function, but it does not
        provide the option of the absolute tolerance.
    """

    A = np.atleast_2d(A)
    s = svd(A, compute_uv=False)
    tol = max(atol, rtol * s[0])
    rank = int((s >= tol).sum())
    return rank


def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


class CyclicSpline(object):

    def __init__(self, knots):
        self.knots = knots
        K = len(self.knots)
        constraints = np.zeros((3 * (K - 1), 4 * (K - 1)))
        for k in range(K - 1):  # loop over the knots
            kp = (k + 1) % (K - 1)
            delta = knots[k + 1] - knots[k]
            for d in range(3):  # loop over the order of the derivates
                constraints[3 * k + d, 4 * kp + d] = - factorial(d)
                for j in range(d, 4):
                    constraints[3 * k + d, 4 * k + j] = (
                        float(factorial(j)) / factorial(j - d)) \
                        * delta ** (j - d)
        self.basis = nullspace(constraints)  # (K-1)*4 , (K-1)

        """Omega matrix of Hastie et al."""
        N = self.basis.shape[1]
        omega = np.zeros((N, N))
        for k in range(len(self.knots) - 1):
            delta = self.knots[k + 1] - self.knots[k]
            for p in range(2, 4):
                for q in range(2, 4):
                    omega += np.outer(self.basis[4 * k + p] * factorial(p),
                                      self.basis[4 * k + q] * factorial(q)
                                      ) * delta ** (p + q - 3)
        self.omega = omega

    def evaluate(self, coefficients, x):
        # Note, assumes x in interval [knots[0], knots[K-1]]

        k = (i for i, _ in enumerate(self.knots)
             if self.knots[i + 1] >= x).next()

        d = x - self.knots[k]
        basis_fxns = np.dot(np.array([d ** p for p in range(4)]),
                            self.basis[(4 * k):(4 * (k + 1))])
        return np.dot(coefficients, basis_fxns)

    def design_matrix(self, x):
        """ Return a design matrix of shape len(x) x basis_dimension """
        matrix = np.zeros((len(x), self.basis.shape[1]))
        for i, v in enumerate(x):
            k = (i for i, _ in enumerate(self.knots)
                 if self.knots[i + 1] >= v).next()
            d = v - self.knots[k]
            basis_fxns = np.dot(np.array([d ** p for p in range(4)]),
                                self.basis[(4 * k):(4 * (k + 1))])
            matrix[i] = basis_fxns
        return matrix

"""
b,k regression
"""


def get_k(theta_k, N_k):
    return np.exp(0.1 * np.dot(N_k, theta_k))  # exp ensures positive k


def vonmises_logpdf(x, kappa):
    return kappa * np.cos(x) - np.log(2 * np.pi * i0(kappa))


def log_likelihood(theta_b, N_b, theta_k, N_k, data):
    kappas = get_k(theta_k, N_k)
    biases = np.dot(N_b, theta_b)
    log_p = 0.
    for k, b, x, y in zip(kappas, biases, data[:, 0], data[:, 1]):
        delta = abs(y - (x + b)) % (2 * np.pi)
        if delta > np.pi:
            delta = 2 * np.pi - delta
        # assert delta <= np.pi
        # assert delta > -np.pi
        log_p += vonmises_logpdf(delta, k)
    return log_p


def prediction_error(theta_b, N_b, data):
    biases = np.dot(N_b, theta_b)
    sse = 0
    for b, x, y in zip(biases, data[:, 0], data[:, 1]):
        delta = abs(y - (x + b)) % (2 * np.pi)
        if delta > np.pi:
            delta = 2 * np.pi - delta
        sse += delta ** 2
    return sse


def fit_transitions(data, knots, penalties, initial_vals=None):

    spline_b = CyclicSpline(knots)
    N_b = spline_b.design_matrix(data[:, 0])
    Omega_b = spline_b.omega

    spline_k = CyclicSpline(knots)
    N_k = spline_k.design_matrix(data[:, 0])
    Omega_k = spline_k.omega

    if initial_vals is None:
        initial_vals = np.zeros(N_b.shape[1] + N_k.shape[1])

    def objective(theta):
        theta_b = theta[:N_b.shape[1]]
        theta_k = theta[N_b.shape[1]:]

        return - log_likelihood(theta_b, N_b, theta_k, N_k, data) + \
            penalties[0] * np.dot(theta_b, np.dot(Omega_b, theta_b)) + \
            penalties[1] * np.dot(theta_k, np.dot(Omega_k, theta_k))

    result = scipy.optimize.minimize(objective, initial_vals, method='BFGS')
    # print result.nfev
    theta = result.x
    return theta[:N_b.shape[1]], N_b, theta[N_b.shape[1]:], N_k, objective(theta)


def fit_biases(data, knots, penalty, initial_vals=None):

    spline_b = CyclicSpline(knots)
    N_b = spline_b.design_matrix(data[:, 0])
    Omega_b = spline_b.omega

    if initial_vals is None:
        initial_vals = np.zeros(N_b.shape[1])

    def objective(theta_b):

        return prediction_error(theta_b, N_b, data) + \
            penalty * np.dot(theta_b, np.dot(Omega_b, theta_b))

    result = scipy.optimize.minimize(objective, initial_vals, method='BFGS')
    # print result.nfev
    theta_b = result.x
    return theta_b, N_b, objective(theta_b)


def fit_kappas(data, knots, penalty, theta_b, initial_vals=None):

    spline_b = CyclicSpline(knots)
    N_b = spline_b.design_matrix(data[:, 0])

    spline_k = CyclicSpline(knots)
    N_k = spline_k.design_matrix(data[:, 0])
    Omega_k = spline_k.omega

    if initial_vals is None:
        initial_vals = np.zeros(N_k.shape[1])

    def objective(theta_k):

        return - log_likelihood(theta_b, N_b, theta_k, N_k, data) + \
            penalty * np.dot(theta_k, np.dot(Omega_k, theta_k))

    result = scipy.optimize.minimize(objective, initial_vals, method='BFGS')
    # print result.nfev
    theta_k = result.x
    return theta_k, N_k, objective(theta_k)


def b_cross_val(data, knots, penalty, repeats=20, single_value_test=True):
    initial_vals = None
    best_err = np.inf

    if isinstance(repeats, int):
        repeat_range = range((len(data)))
        repeats = [np.random.permutation(repeat_range) for _ in range(repeats)]

    errs, bs, = [], []
    for indices in repeats:
        if single_value_test:
            trainIdx = indices[:-1]
            testIdx = [indices[-1]]
        else:
            trainIdx = indices[::2]
            testIdx = indices[1::2]
        training = data[trainIdx]
        testing = data[testIdx]
        theta_b, N_b, _ = fit_biases(
            training, knots, penalty, initial_vals=initial_vals)
        err = prediction_error(theta_b, N_b, testing)
        errs.append(err)
        b = np.dot(N_b, theta_b)
        bs.append([x for (y, x) in sorted(zip(
                  indices, np.hstack([b, np.full(len(testIdx), np.nan)])))])
        if err < best_err:
            best_err = err
            initial_vals = theta_b

    return np.mean(errs), np.nanmean(bs, axis=0)


def k_cross_val(data, knots, penalty, theta_b, repeats=20, single_value_test=True):
    initial_vals = None
    best_logL = -np.inf

    spline_b = CyclicSpline(knots)
    N_b = spline_b.design_matrix(data[:, 0])

    if isinstance(repeats, int):
        repeat_range = range((len(data)))
        repeats = [np.random.permutation(repeat_range) for _ in range(repeats)]

    logLs, ks = [], []
    for indices in repeats:
        if single_value_test:
            trainIdx = indices[:-1]
            testIdx = [indices[-1]]
        else:
            trainIdx = indices[::2]
            testIdx = indices[1::2]
        training = data[trainIdx]
        testing = data[testIdx]
        theta_k, N_k, _ = fit_kappas(
            training, knots, penalty, theta_b, initial_vals=initial_vals)
        ll = log_likelihood(theta_b, N_b, theta_k, N_k, testing)
        logLs.append(ll)
        k = get_k(theta_k, N_k)
        ks.append([x for (y, x) in sorted(zip(indices, np.hstack([k, np.full(len(testIdx), np.nan)])))])
        if ll > best_logL:
            best_logL = ll
            initial_vals = theta_k

    return np.mean(logLs), np.nanmean(ks, axis=0)


def bk_cross_val(data, knots, penalties, repeats=20, single_value_test=True):

    initial_vals = None
    best_logL = -np.inf

    if isinstance(repeats, int):
        repeat_range = range((len(data)))
        repeats = [np.random.permutation(repeat_range) for _ in range(repeats)]

    logLs, bs, ks = [], [], []
    for indices in repeats:
        if single_value_test:
            trainIdx = indices[:-1]
            testIdx = [indices[-1]]
        else:
            trainIdx = indices[::2]
            testIdx = indices[1::2]
        training = data[trainIdx]
        testing = data[testIdx]
        theta_b, N_b, theta_k, N_k, _ = fit_transitions(
            training, knots, penalties, initial_vals=initial_vals)
        ll = log_likelihood(theta_b, N_b, theta_k, N_k, testing)
        logLs.append(ll)
        b = np.dot(N_b, theta_b)
        bs.append([x for (y, x) in sorted(zip(indices, np.hstack([b, np.full(len(testIdx), np.nan)])))])
        k = get_k(theta_k, N_k)
        ks.append([x for (y, x) in sorted(zip(indices, np.hstack([k, np.full(len(testIdx), np.nan)])))])
        if ll > best_logL:
            best_logL = ll
            initial_vals = np.hstack([theta_b, theta_k])

    return np.mean(logLs), np.nanmean(bs, axis=0), np.nanmean(ks, axis=0)


"""
Logistic regression
"""


def prob(theta, N):
    f_x = np.dot(N, theta)
    return 1. / (1. + np.exp(-f_x))


def update(theta, N, Omega, lambda_, y):
    lOmega = lambda_ * Omega

    p = prob(theta, N)
    w = p * (1 - p)
    wz = w * np.dot(N, theta) + (y - p)
    return np.dot(np.linalg.inv(np.dot(N.T * w, N) + lOmega),
                  np.dot(N.T, wz))


def fit_model(y, N, Omega, knots, penalty, max_iter=1000, thresh=1e-12):
    """
    Data has format (n, 2), where data[n] = (x, y) with y = 0, 1
    """
    theta = np.zeros(len(Omega))
    for i in range(max_iter):
        theta_new = update(theta, N, Omega, penalty, y)
        norm = np.linalg.norm(theta - theta_new)
        theta = theta_new
        if norm < thresh:
            return theta
    print 'No convergence', norm
    return theta


def nonparameteric_logistic_regression(
        data, knots, penalty, max_iter=1000, thresh=1e-12, repeats=20,
        single_value_test=False):

    spline = CyclicSpline(knots)
    N = spline.design_matrix(data[:, 0])
    Omega = spline.omega

    fits = []
    logLs = []

    if isinstance(repeats, int):
        repeat_range = range(len(data))
        repeats = [np.random.permutation(repeat_range) for _ in range(repeats)]
    for indices in repeats:
        if single_value_test:
            trainIdx = indices[:-1]
            testIdx = [indices[-1]]
        else:
            trainIdx = indices[::2]
            testIdx = indices[1::2]
        training = data[trainIdx]
        testing = data[testIdx]

        theta = fit_model(training[:, 1], N[trainIdx], Omega, knots, penalty,
                          max_iter=max_iter, thresh=thresh)
        cross_logL = np.log(prob(theta, N[testIdx])[testing[:, 1] > 0.5]).sum() + \
            np.log(1 - prob(theta, N[testIdx])[testing[:, 1] < 0.5]).sum()
        fits.append(prob(theta, N))
        logLs.append(cross_logL)
    return fits, np.mean(logLs)


def test_logistic_regression():
    K = 20
    knots = np.linspace(-np.pi, np.pi, K)
    # sp = CyclicSpline(knots)
    # print sp.basis.shape
    D = 100
    data = np.zeros((D, 2))
    data[:, 0] = sorted(-np.pi + 2 * np.pi * np.random.rand(D))
    data[:, 1] = np.random.randint(2, size=D)
    penalty = 0.25 * D

    # c = np.random.randn(K-1)
    # x = np.linspace(-np.pi, np.pi, 100)
    # y = [sp.evaluate(c, p) for p in x]
    #
    # dm = sp.design_matrix(x)

    # plt.imshow(sp.omega, interpolation='none')
    # plt.show()
    # plt.figure()
    # plt.plot(x, y)
    # plt.show()

    D = 100
    data = np.zeros((D, 2))
    data[:, 0] = sorted(-np.pi + 2 * np.pi * np.random.rand(D))
    data[:, 1] = np.random.randint(2, size=D)

    for penalty in np.exp(range(-5, 5)):

        fits, logL = nonparameteric_logistic_regression(data, knots, penalty)
        # plt.plot(data[:, 0], np.mean(fits, axis=0))
        plt.plot(data[:, 0], fits[0])
        print penalty, logL

    plt.plot(data[:, 0], data[:, 1], 'ko')
    plt.show()

    # c = np.random.randn(K-1)
    # x = np.linspace(-np.pi, np.pi, 100)
    # y = [sp.evaluate(c, p) for p in x]
    #
    # dm = sp.design_matrix(x)

    # plt.imshow(sp.omega, interpolation='none')
    # plt.show()
    # plt.figure()
    # plt.plot(x, y)
    # plt.show()


    # follow Hastie et al.

    """
    COMPUTE N

    N_j are the natural spline bases

    {N}_ij = N_j(x_i)

    Omega_jk = \int N''_j(t)N''_k(t)dt
    n = len(data)
    """

# def test_transition_fit():


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    # test_transition_fit()
    K = 20
    knots = np.linspace(-np.pi, np.pi, K)
    D = 500
    data = -np.pi + 2 * np.pi * np.random.rand(D, 2)
    data[:, 1] = data[:, 0] + 0.1 * np.sin(data[:, 0]) + 0.1 * np.random.randn(D)
    data[:, 1] = ((data[:, 1] + np.pi) % (2*np.pi)) - np.pi
    theta_b, N_b, theta_k, N_k, ll = fit_transitions(data, knots, [1., 1.])
    print ll
    b = np.dot(N_b, theta_b)
    plt.plot(data[:, 0], b, '.')
    plt.figure()
    plt.plot(data[:, 0], get_k(theta_k, N_k), '.')
    plt.show()

    # test_logistic_regression()
