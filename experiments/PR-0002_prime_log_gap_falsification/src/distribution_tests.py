import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings


def ks_test_normal(data: np.ndarray):
    """
    KS test against normal distribution.
    """
    # Fit normal
    mu, sigma = stats.norm.fit(data)
    ks_stat, p_value = stats.kstest(data, "norm", args=(mu, sigma))
    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "params": {"mu": mu, "sigma": sigma},
    }


def ks_test_lognormal(data: np.ndarray):
    """
    KS test against log-normal distribution.
    """
    # Fit log-normal
    shape, loc, scale = stats.lognorm.fit(data, floc=0)  # fix loc=0
    ks_stat, p_value = stats.kstest(data, "lognorm", args=(shape, loc, scale))
    # MLE for log-normal parameters
    log_data = np.log(data)
    mu_mle = np.mean(log_data)
    sigma_mle = np.std(log_data)
    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "params": {"shape": shape, "loc": loc, "scale": scale},
        "mle_params": {"mu": mu_mle, "sigma": sigma_mle},
    }


def ks_test_exponential(data: np.ndarray):
    """
    KS test against exponential distribution.
    """
    # Fit exponential
    loc, scale = stats.expon.fit(data, floc=0)  # fix loc=0
    ks_stat, p_value = stats.kstest(data, "expon", args=(loc, scale))
    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "params": {"loc": loc, "scale": scale},
    }


def ks_test_gamma(data: np.ndarray):
    """
    KS test against gamma distribution.
    """
    # Fit gamma
    a, loc, scale = stats.gamma.fit(data, floc=0)
    ks_stat, p_value = stats.kstest(data, "gamma", args=(a, loc, scale))
    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "params": {"a": a, "loc": loc, "scale": scale},
    }


def ks_test_weibull(data: np.ndarray):
    """
    KS test against Weibull distribution.
    """
    # Fit Weibull
    c, loc, scale = stats.weibull_min.fit(data, floc=0)
    ks_stat, p_value = stats.kstest(data, "weibull_min", args=(c, loc, scale))
    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "params": {"c": c, "loc": loc, "scale": scale},
    }


def ks_test_uniform(data: np.ndarray):
    """
    KS test against uniform distribution (for falsification).
    """
    # Fit uniform
    loc, scale = stats.uniform.fit(data)
    ks_stat, p_value = stats.kstest(data, "uniform", args=(loc, scale))
    return {
        "ks_stat": ks_stat,
        "p_value": p_value,
        "params": {"loc": loc, "scale": scale},
    }


def run_distribution_tests(data: np.ndarray):
    """
    Run all KS tests and return results.
    """
    tests = {
        "normal": ks_test_normal(data),
        "lognormal": ks_test_lognormal(data),
        "exponential": ks_test_exponential(data),
        "gamma": ks_test_gamma(data),
        "weibull": ks_test_weibull(data),
        "uniform": ks_test_uniform(data),
    }
    return tests


def find_best_fit(tests: dict):
    """
    Find the distribution with the smallest KS statistic.
    """
    best_dist = min(tests, key=lambda x: tests[x]["ks_stat"])
    return best_dist, tests[best_dist]["ks_stat"]


if __name__ == "__main__":
    # Test with small data
    np.random.seed(42)
    data = np.random.lognormal(0, 1, 100)
    tests = run_distribution_tests(data)
    print("Best fit:", find_best_fit(tests))
    for dist, result in tests.items():
        print(f"{dist}: KS={result['ks_stat']:.4f}, p={result['p_value']:.4f}")
