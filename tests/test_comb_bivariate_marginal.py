"""
Test whether the marginal probabilities of bivariate distribution match univariate probabilities.
(Purely analytical identity checks, no simulation involved)
"""

import random
import pytest
from occenv.comb_bivariate import CombinatorialBivariate
from occenv.comb_univariate import CombinatorialUnivariate


@pytest.mark.parametrize(
    "shard_sizes",
    [
        (50,),
        (50, 40),
        (50, 50, 40),
        (50, 50, 50, 50),
    ],
)
def test_bivariate_marginals_match_univariate(shard_sizes):
    """
    Test whether the marginal probabilities of bivariate distribution match univariate probabilities.
    (Purely analytical identity checks, no simulation involved)
    """
    total_number = 200
    comb = CombinatorialBivariate(total_number, shard_sizes)
    comb_univariate = CombinatorialUnivariate(total_number, shard_sizes)

    # check a few v values (marginal probability conditioned on union)
    intersection_values = [
        0,
        min(shard_sizes) // 2,
        min(shard_sizes),
        random.randint(0, min(shard_sizes)),
    ]
    for v in intersection_values:
        sum_marginal_prob = sum(
            comb.bivariate_prob(u, v) for u in range(0, total_number + 1)
        )
        comb_intersection_prob = comb_univariate.intersection_prob(v)
        assert comb_intersection_prob == pytest.approx(sum_marginal_prob, abs=1e-10)

    # check a few u values (marginal probability conditioned on intersection)
    union_values = [
        max(shard_sizes),
        (max(shard_sizes) + total_number) // 2,
        total_number,
        random.randint(0, max(shard_sizes)),
    ]
    for u in union_values:
        sum_marginal_prob = sum(
            comb.bivariate_prob(u, v) for v in range(0, min(shard_sizes) + 1)
        )
        comb_union_prob = comb_univariate.union_prob(u)
        assert comb_union_prob == pytest.approx(sum_marginal_prob, abs=1e-10)
