
import numpy as np
import math
from scipy.stats import norm
from scipy.integrate import quad
from math import factorial


def compute_delta(b, n, q, gap_q, distribution = None, verbose = False, sigma = None):

    """
    Computes the maximum delta (approximate differential privacy parameter) value for the given parameters, using different 
    bounds (e.g., Hoeffding and Bernstein inequalities) depending on the distribution (uniform or truncated normal).

    Parameters:
    - b (float): The range [0, b] for the samples or the upper bound of the interval.
    - n (int): The total number of samples.
    - q (float): The quantile (0, 1) used for calculations.
    - gap_q (float): The minimum gap required between consecutive samples.
    - distribution (str, optional): Specifies the distribution of the data. 
      Options include "uniform" and "truncated_normal".
    - verbose (bool, optional): If True, prints additional debugging information.
    - sigma (float, optional): The standard deviation of the truncated normal distribution (required if `distribution='truncated_normal'`).

    Returns:
    - float: The computed maximum delta value, ensuring appropriate bounds are met.

    Functionality:
    - For `uniform` distribution, uses Hoeffding and Bernstein inequalities to calculate delta.
    - For `truncated_normal` distribution, uses a Hoeffding-style bound with adjustments for expected order statistics.
    """


    rs = [math.ceil(q*n) - 1, math.ceil(q*n)]
    final_delta = 0
    if distribution == 'uniform':

        d = (gap_q  - 2*b / (n+1)) / 2 * b
        
        assert d > 0, f"{(gap_q, 2*b / (n+1))}"   # this allows us to use a beta distribution as the distribution of the order statistic (U(i) - a) / (b- a)

        # we use both hoeffding and bernstein inequalities to pick the most suitable bound

        for r in rs:

            var1_bern = (r * (n - r + 1)) / ((n + 1)**2 * (n+2))
            var2_bern = ((r+1) * (n - r)) / ((n + 1)**2 * (n+2))

            num = - d**2
            denom1 = (var1_bern + b * d / 3)
            denom2 = (var2_bern + b * d / 3)

            _delta_bern = 2 * np.exp( num / denom1) + 2 * np.exp(num / denom2)
            _delta_hoeff = 4 * np.exp( - 2 * d ** 2)

            delta = min(_delta_bern, _delta_hoeff)
            final_delta = max(final_delta, delta)


    elif distribution == 'truncated_normal':
        assert sigma != None

        for r in rs:
            d = (gap_q - expected_order_stat(i=r, n=n, b=b, sigma=sigma)) / 2
            assert d >= 0, f"{(gap_q, expected_order_stat(i=r, n=n, b=b, sigma=sigma))}"
            _delta_hoeff = 4 * np.exp( - d ** 2 / (2 * b ** 2))
            final_delta = max(final_delta, _delta_hoeff)
    
    if verbose:

        print(('MAX DELTA', final_delta))

    return final_delta


def truncated_normal_pdf(x, b, sigma=1):
    """
    PDF of a truncated normal distribution with mean 0 and standard deviation sigma on [-b, b].
    
    Parameters:
    x : float
        The point at which to evaluate the PDF.
    b : float
        The truncation boundary.
    sigma : float
        The standard deviation of the normal distribution.
        
    Returns:
    float
        The value of the PDF at x.
    """
    Z = norm.cdf(b / sigma) - norm.cdf(-b / sigma)  # Normalizing constant
    return norm.pdf(x / sigma) / (sigma * Z) if -b <= x <= b else 0

def truncated_order_stat_pdf(x, i, n, b, sigma=1):
    """
    PDF of the i-th order statistic for n i.i.d. truncated normal variables.
    
    Parameters:
    x : float
        The point at which to evaluate the PDF.
    i : int
        The index of the order statistic (1-based).
    n : int
        The total number of random variables.
    b : float
        The truncation boundary.
    sigma : float
        The standard deviation of the normal distribution.
        
    Returns:
    float
        The value of the PDF at x.
    """
    F_x = quad(lambda t: truncated_normal_pdf(t, b, sigma), -b, x)[0]  # CDF for truncated normal
    f_x = truncated_normal_pdf(x, b, sigma)  # PDF for truncated normal
    coeff = factorial(n) / (factorial(i - 1) * factorial(n - i))
    return coeff * (F_x**(i - 1)) * f_x * ((1 - F_x)**(n - i))

def expected_order_stat(i, n, b, sigma=1):
    """
    Computes the expected value of the i-th order statistic for n i.i.d. truncated normal variables.
    
    Parameters:
    i : int
        The index of the order statistic (1-based).
    n : int
        The total number of random variables.
    b : float
        The truncation boundary.
    sigma : float
        The standard deviation of the normal distribution.
        
    Returns:
    float
        The expected value of the i-th order statistic.
    """
    # Define the integrand
    integrand = lambda x: x * truncated_order_stat_pdf(x, i, n, b, sigma)
    
    # Compute the integral over the truncated domain
    result, _ = quad(integrand, -b, b)
    return result
