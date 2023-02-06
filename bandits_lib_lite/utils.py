import numpy as np
from scipy.stats import norm, rv_continuous


def inv_sherman_morrison(u, A_inv):
    """Inverse of a matrix with rank 1 update."""
    Au = np.dot(A_inv, u)
    A_inv -= np.outer(Au, Au) / (1 + np.dot(u.T, Au))
    return A_inv


def expectile_loss(q):
    return lambda x: (1 - q) * x**2 * (x < 0) + q * x**2 * (x > 0)


def get_gaussian_expectile(q, mu=0.0, sigma=1.0, n_quadrature=25):
    """q-expectile in the Gaussian case using Gauss-Hermite quadrature
    to speed things up.
    """
    from scipy.special import roots_hermitenorm
    from scipy.optimize import minimize_scalar

    knots, weights = roots_hermitenorm(n_quadrature)
    loss = expectile_loss(q)
    obj = lambda x: np.dot(
        list(map(lambda u: loss(sigma * u + mu - x), knots)), weights
    )
    res = minimize_scalar(obj)
    if res.success:
        return res.x
    else:
        raise Exception("Expectile calculation failed...")


def get_expectile(q, distr, low=-np.inf, high=np.inf):
    """Generic q-expectile calculator.
    Pros: generic. Cons: sloooow.
    """
    from scipy.optimize import minimize_scalar
    from scipy.integrate import quad

    loss = expectile_loss(q)

    def obj(x):
        integrand = lambda y: loss(y - x) * distr.pdf(y)
        ret, _ = quad(integrand, low, high)
        return ret

    res = minimize_scalar(obj)
    if res.success:
        return res.x
    else:
        raise Exception("Expectile calculation failed...")


def get_empirical_expectile(q, sample):
    """q-expectile of a given discrete sample."""
    from scipy.optimize import minimize_scalar

    loss = expectile_loss(q)
    obj = lambda x: np.mean(list(map(lambda y: loss(y - x), sample)))
    res = minimize_scalar(obj)
    if res.success:
        return res.x
    else:
        raise Exception("Expectile calculation failed...")


def expectile_regret_helper(bandit, features, action, q, gaussian_noise):
    if gaussian_noise:
        if bandit._shared_noise:
            mus = np.array(bandit.mus(features)) + np.array(
                [bandit.noise_dist.mean() for _ in bandit.arms]
            )
            sigmas = [bandit.noise_dist.std() for _ in bandit.arms]
        else:
            mus = np.array(bandit.mus(features)) + np.array(
                [bandit.noise_dist[k].mean() for k in bandit.arms]
            )
            sigmas = [bandit.noise_dist[k].std() for k in bandit.arms]

        expectiles = [
            get_gaussian_expectile(
                q,
                mus[k],
                sigmas[k],
            )
            for k in bandit.arms
        ]
        return np.max(expectiles) - expectiles[action]
    else:
        raise NotImplementedError(
            "Only do expectile regret for Gaussian noise for now."
        )


def make_gaussian_with_given_expectile(q, expectile, sigma, n_quadrature=25):
    return norm(
        loc=expectile
        - get_gaussian_expectile(q, mu=0.0, sigma=sigma, n_quadrature=n_quadrature),
        scale=sigma,
    )


def get_bernoulli_entropic_risk(gamma, p, a, b):
    """Entropic Risk 1 / gamma * log E[e^{gamma * Y}]
    where Y = a with proba p and Y = b with proba 1 - p.
    """
    return 1 / gamma * np.log(p * np.exp(gamma * a) + (1 - p) * np.exp(gamma * b))


class AsymmetricNorm(rv_continuous):
    """
    Asymmetric Gaussian distribution, with zero expectile.
    """

    def __init__(self, q=0.5, loc=0.0, scale=1.0, **kwargs):
        self.q = q
        self.loc = loc
        self.scale = scale
        self.C = (
            np.sqrt(2 * self.q * (1 - self.q) / np.pi)
            / (np.sqrt(self.q) + np.sqrt(1 - self.q))
            / self.scale
        )
        super().__init__(**kwargs)

    def _pdf(self, x):
        return self.C * np.exp(
            -0.5 / self.scale**2
            * np.abs(self.q - (x - self.loc < 0))
            * (x - self.loc) ** 2
        )
