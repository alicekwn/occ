"""
Test the CLT approximated results vs analytical (combinatorial) results.
Testing whether the mean and variance matches.
"""

import pytest
from occenv.clt_special_case import CltSpecialCase
from occenv.comb_univariate import CombinatorialUnivariate
from occenv.comb_bivariate import CombinatorialBivariate
from occenv.comb_jaccard import CombinatorialJaccard


@pytest.mark.parametrize(
    "total_number, shard_sizes",
    [
        (100, (10, 20, 30)),
        (100, (10, 20, 30, 40)),
        (100, (50, 50)),
        (200, (100, 100, 100)),
        (200, (10, 10, 8, 5)),  # edge case
        (200, (190, 180, 195, 199)),  # edge case
    ],
)
def test_approx(total_number, shard_sizes):
    """
    Test that the (CLT) approximated mean and variance match the analytical mean and variance.
    """
    univ = CombinatorialUnivariate(total_number, shard_sizes)
    biv = CombinatorialBivariate(total_number, shard_sizes)
    jaccard = CombinatorialJaccard(total_number, shard_sizes, biv)
    approx_result = CltSpecialCase(total_number, shard_sizes)

    # ----- Union distribution -------
    p_union_approx = approx_result.union_p_approx()

    assert approx_result.mu_sum_indicators(p_union_approx) == pytest.approx(
        univ.union_mu(), abs=0.01
    )
    assert approx_result.union_var_approx() == pytest.approx(univ.union_var(), abs=0.01)

    # ----- Intersection distribution -------
    p_intersection_approx = approx_result.intersection_p_approx()
    assert approx_result.mu_sum_indicators(p_intersection_approx) == pytest.approx(
        univ.intersection_mu(), abs=0.01
    )
    assert approx_result.intersection_var_approx() == pytest.approx(
        univ.intersection_var(), abs=0.01
    )

    # ----- Bivariate distribution -------
    assert approx_result.bivariate_mu_approx() == pytest.approx(
        biv.bivariate_mu(), abs=0.01
    )
    assert approx_result.bivariate_matrix_approx() == pytest.approx(
        biv.bivariate_matrix(), abs=0.01
    )

    # ----- Jaccard index distribution -------
    assert approx_result.jaccard_mu_approx() == pytest.approx(
        jaccard.jaccard_mu(), abs=0.01
    )
    assert approx_result.jaccard_var_approx() == pytest.approx(
        jaccard.jaccard_var(), abs=0.01
    )
