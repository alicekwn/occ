"""
Test the distribution of X=LZ against simulation.
Z is degree-count vector and L is a vector of weights (any real numbers).
"""

import numpy as np
import pytest
from scipy.stats import norm

from occenv.clt_degree_approx import CltDegreeVector
from occenv.simulate import Simulate


@pytest.mark.parametrize(
    "total_number, shard_sizes, weights, repeats, tolerance",
    [
        (100, (40, 50, 60, 70), [-2, 3.1, 2, 4.2, -1.5], 100000, 0.01),
        (1000, (400, 500, 600, 700), [-2, 3.1, 2, 4.2, -1.5], 100000, 0.01),
    ],
)
def test_pdf_integral_matches_pmf_sum(
    total_number, shard_sizes, weights, repeats, tolerance
):
    """
    Test that the PDF integral from μ-σ/4 to μ+σ/4 matches the PMF sum in that range.
    The PDF integral ∫[μ-σ/4 to μ+σ/4] f(x) dx should approximate
    the PMF sum P(μ-σ/4 < X ≤ μ+σ/4) from simulation.
    """
    # Get CLT approximation parameters
    approx = CltDegreeVector(total_number, shard_sizes)
    mu = approx.z_means_weighted(weights)
    var = approx.z_vars_weighted(weights)
    sigma = np.sqrt(var) if var > 0 else 0.0

    if sigma == 0:
        pytest.skip("Variance is zero, cannot test with σ=0")

    # Compute PDF integral from μ-σ/2 to μ+σ/2
    pdf_integral = norm.cdf(mu + sigma / 4, loc=mu, scale=sigma) - norm.cdf(
        mu - sigma / 4, loc=mu, scale=sigma
    )

    # Simulate and compute PMF sum in range (μ-σ/4, μ+σ/4]
    sim = Simulate(total_number, shard_sizes)
    degree_counts = sim.simulate_degree_count_repeat(repeats)
    t_counts = degree_counts.dot(weights)

    # Count values in range (μ-σ/4, μ+σ/4]
    in_range = (t_counts > mu - sigma / 4) & (t_counts <= mu + sigma / 4)
    pmf_sum = np.sum(in_range) / repeats

    # Compare PDF integral with PMF sum
    difference = abs(pdf_integral - pmf_sum)
    assert (
        difference < tolerance
    ), f"PDF integral ({pdf_integral:.4f}) and PMF sum ({pmf_sum:.4f}) differ by {difference:.4f}, exceeding tolerance {tolerance}"
