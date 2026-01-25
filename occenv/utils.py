"""
Utility functions for calculating statistical measures of discrete distributions.
"""

import numpy as np
import scipy.stats as stats
from scipy.stats import norm, multivariate_normal


def mu_calculation(x_values: list[float], pmf: list[float]) -> float:
    """
    Statistical mean of a discrete distribution.
    """
    x_arr = np.array(x_values, dtype=float)
    p_arr = np.array(pmf, dtype=float)
    return float(np.sum(x_arr * p_arr))


def var_calculation(x_values: list[float], pmf: list[float]) -> float:
    """
    Statistical variance of a discrete distribution.
    """
    mu = mu_calculation(x_values, pmf)
    x_arr = np.array(x_values, dtype=float)
    p_arr = np.array(pmf, dtype=float)
    return float(np.sum((x_arr - mu) ** 2 * p_arr))


def sd_calculation(x_values: list[float], pmf: list[float]) -> float:
    """
    Statistical standard deviation of a discrete distribution.
    """
    return np.sqrt(var_calculation(x_values, pmf))


def mae_calculation(normal_pdf: list[float], approx_pdf: list[float]) -> float:
    """
    Mean absolute error between two lists.
    """
    normal_pdf_arr = np.array(normal_pdf, dtype=float)
    approx_pdf_arr = np.array(approx_pdf, dtype=float)
    return np.mean(np.abs(normal_pdf_arr - approx_pdf_arr))


def sse_calculation(normal_pdf: list[float], approx_pdf: list[float]) -> float:
    """
    Sum of squared errors between two lists.
    """
    normal_pdf_arr = np.array(normal_pdf, dtype=float)
    approx_pdf_arr = np.array(approx_pdf, dtype=float)
    return np.sum((normal_pdf_arr - approx_pdf_arr) ** 2)


def norm_pdf(x: float, mu: float, sigma: float) -> float:
    """
    Normal probability density function (continuous).
    """
    if sigma is None or sigma <= 0:
        return np.zeros_like(x, dtype=float)
    return stats.norm.pdf(x, mu, sigma)


def discretize_normal_pmf(x_vals: list[float], mu: float, sigma: float) -> list[float]:
    """
    Discretize a normal probability density function (continuous) into a discrete distribution.
    """
    x = np.asarray(sorted(x_vals), dtype=float)
    if sigma is None or sigma <= 0:
        return np.zeros_like(x)
    # Calculate cell boundaries (midpoints)
    mids = (x[1:] + x[:-1]) / 2
    edges = np.concatenate(([-np.inf], mids, [np.inf]))
    return np.diff(
        norm.cdf(edges, loc=mu, scale=sigma)
    )  # Calculate mass per cell = CDF difference


def discretize_bivariate_normal_pmf(
    u_vals: list[int], v_vals: list[int], mu: np.ndarray, cov: np.ndarray
) -> list[float]:
    """
    Discretize a bivariate normal distribution over a 2D grid.

    Args:
        u_vals: List of u (union) values
        v_vals: List of v (intersection) values
        mu: Mean vector [mu_u, mu_v]
        cov: Covariance matrix [[var_u, cov_uv], [cov_uv, var_v]]

    Returns:
        Flattened PMF as a list, matching the order:
        [P(u0,v0), P(u0,v1), ..., P(u0,vn), P(u1,v0), ...]
    """
    u_vals = np.asarray(sorted(u_vals), dtype=float)
    v_vals = np.asarray(sorted(v_vals), dtype=float)
    mu = np.asarray(mu, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    n_total = len(u_vals) * len(v_vals)

    eigvals = np.linalg.eigvals(cov)
    if (
        np.any(eigvals <= 0)
        or np.any(np.isnan(eigvals) | np.isinf(eigvals))
        or np.min(eigvals) < 1e-10
    ):
        return [0.0] * n_total
    mvn = multivariate_normal(mean=mu, cov=cov)

    # Calculate cell boundaries (midpoints between consecutive values)
    def get_edges(vals):
        if len(vals) > 1:
            mids = (vals[1:] + vals[:-1]) / 2
            return np.concatenate(([-np.inf], mids, [np.inf]))
        return np.array([-np.inf, vals[0], np.inf])

    u_edges, v_edges = get_edges(u_vals), get_edges(v_vals)

    # Compute probability mass for each cell
    pmf_2d = np.zeros((len(v_vals), len(u_vals)), dtype=float)
    for i in range(len(v_vals)):
        for j in range(len(u_vals)):
            u_low, u_high = u_edges[j], u_edges[j + 1]
            v_low, v_high = v_edges[i], v_edges[i + 1]
            cell_area = (u_high - u_low) * (v_high - v_low)

            if np.isnan(cell_area) or np.isinf(cell_area):
                continue

            try:
                pdf_value = mvn.pdf([(u_low + u_high) / 2, (v_low + v_high) / 2])
                if not (np.isnan(pdf_value) or np.isinf(pdf_value)):
                    pmf_2d[i, j] = pdf_value * cell_area
            except (ValueError, np.linalg.LinAlgError):
                pass

    # Normalize
    total = pmf_2d.sum()
    if total > 0 and not (np.isnan(total) or np.isinf(total)):
        pmf_2d /= total
    else:
        pmf_2d.fill(0.0)

    return pmf_2d.flatten(order="C").tolist()
