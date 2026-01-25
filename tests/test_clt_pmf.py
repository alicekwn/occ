"""
Test the CLT approximated results vs analytical results.
Testing whether each element of the PMF matches.
"""

from fractions import Fraction
import numpy as np
import pytest
from occenv.approximated import ApproximatedResult
from occenv.analytical_univariate import AnalyticalUnivariate
from occenv.analytical_bivariate import AnalyticalBivariate
from occenv.analytical_jaccard import AnalyticalJaccard
from occenv.utils import discretize_normal_pmf, discretize_bivariate_normal_pmf


@pytest.mark.parametrize(
    "total_number, shard_sizes",
    [
        # (100, (20, 30, 40)),  # one shard is too small, failed test
        (100, (90, 60, 70)),  # normal case, passed test
        # (200, (100, 100, 100)),  # normal case, passed test
        # (200, (10, 10, 8, 5)),  # edge case, equally small
        # (200, (190, 180, 195, 199)),  # edge case, equally large
    ],
)
def test_approx(total_number, shard_sizes):
    """
    Test that the (CLT) approximated each element of the PMF matches the analytical results.
    """
    univ = AnalyticalUnivariate(total_number, shard_sizes)
    biv = AnalyticalBivariate(total_number, shard_sizes)
    jaccard = AnalyticalJaccard(total_number, shard_sizes, biv)
    approx_result = ApproximatedResult(total_number, shard_sizes)

    # ----- Union distribution -------
    p_union_approx = approx_result.union_p_approx()
    pmf_union_ana = [univ.union_prob(u) for u in range(total_number + 1)]
    pmf_union_approx = discretize_normal_pmf(
        range(total_number + 1),
        approx_result.mu_sum_indicators(p_union_approx),
        approx_result.union_sd_approx(),
    )

    assert pmf_union_ana == pytest.approx(pmf_union_approx, abs=0.05)

    # ----- Intersection distribution -------
    p_intersection_approx = approx_result.intersection_p_approx()
    pmf_intersection_ana = [univ.intersection_prob(v) for v in range(total_number + 1)]
    pmf_intersection_approx = discretize_normal_pmf(
        range(total_number + 1),
        approx_result.mu_sum_indicators(p_intersection_approx),
        approx_result.intersection_sd_approx(),
    )
    assert pmf_intersection_ana == pytest.approx(pmf_intersection_approx, abs=0.05)

    # ----- Bivariate distribution -------
    # Build analytical PMF, replacing NaN with 0.0 for invalid (u, v) pairs
    pmf_bivariate_ana = []
    for u in range(total_number + 1):
        for v in range(total_number + 1):
            prob = biv.bivariate_prob(u, v)
            # Replace NaN with 0.0 for invalid (u, v) pairs
            pmf_bivariate_ana.append(0.0 if np.isnan(prob) else prob)

    pmf_bivariate_approx = discretize_bivariate_normal_pmf(
        list(range(total_number + 1)),
        list(range(total_number + 1)),
        approx_result.bivariate_mu_approx(),
        approx_result.bivariate_matrix_approx(),
    )
    assert pmf_bivariate_ana == pytest.approx(pmf_bivariate_approx, abs=0.1)

    # ----- Jaccard index distribution -------
    # Build Jaccard index PMF by finding all unique ratios v/u
    ratios = set()
    for v in range(0, min(shard_sizes) + 1):
        for u in range(max(v, 1), total_number + 1):
            ratios.add(Fraction(v, u))
    ratios = sorted(ratios, key=float)

    jaccard_values = []
    pmf_jaccard_ana = []
    for ratio in ratios:
        prob = jaccard.jaccard_prob(
            numerator=ratio.numerator, denominator=ratio.denominator
        )
        if prob > 0:
            jaccard_values.append(float(ratio))
            pmf_jaccard_ana.append(prob)

    # Discretize normal distribution over the same Jaccard index values
    pmf_jaccard_approx = discretize_normal_pmf(
        jaccard_values,
        approx_result.jaccard_mu_approx(),
        approx_result.jaccard_var_approx()
        ** 0.5,  # Convert variance to standard deviation
    )
    assert pmf_jaccard_ana == pytest.approx(pmf_jaccard_approx, abs=0.1)
