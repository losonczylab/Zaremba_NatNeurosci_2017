import numpy as np
from scipy.stats import vonmises
import matplotlib.pyplot as plt


def density(x, data, kappa):
    denom = np.exp(kappa) / vonmises.pdf(0., kappa)
    N = len(data)
    p = 0.
    for z in data:
        p += np.exp(np.cos(x - z) * kappa) / denom / N
    return p


def cross_val(kappa, data, repeats=1, single_value_test=False):
    log_likelihoods = []
    if isinstance(repeats, int):
        repeats = [np.random.permutation(range(len(data))) for _ in range(repeats)]
    for indices in repeats:
        if single_value_test:
            trainIdx = indices[:-1]
            testIdx = [indices[-1]]
        else:
            trainIdx = indices[::2]
            testIdx = indices[1::2]
        training = data[trainIdx]
        testing = data[testIdx]
        log_likelihoods.append(
            sum(np.log(density(x, training, kappa)) for x in testing))
    return np.mean(log_likelihoods)


if __name__ == '__main__':
    N = 100
    data = -np.pi + 2 * np.pi * np.random.rand(N)

    x = np.linspace(-np.pi, np.pi, 200)
    for kappa in np.exp(np.linspace(-3, 3, 8)):
        print kappa, cross_val(kappa, data)
        y = [density(v, data, kappa) for v in x]
        plt.plot(x, y)
    plt.show()
