"""
Test that the combinatorial Jaccard index CDF matches the simulated Jaccard index CDF.
"""

import pytest
from occenv.simulate import Simulate
from occenv.comb_jaccard import CombinatorialJaccard
from occenv.comb_bivariate import CombinatorialBivariate


@pytest.mark.parametrize(
    "total_number, shard_sizes, thresholds",
    [
        (100, (30, 35), [0.2, 0.3, 0.4, 0.5]),
        (80, (20, 25), [0.25, 0.4, 0.5]),
        (60, (15, 20), [0.2, 0.33, 0.5]),
    ],
)
def test_jaccard_cdf(total_number, shard_sizes, thresholds):
    """
    Test that the combinatorial Jaccard index CDF matches the simulated Jaccard index CDF.
    """
    repeats = int(1e6)
    sim = Simulate(total_number, shard_sizes)
    repeat_degree_counts = sim.simulate_degree_count_repeat(repeat=repeats)
    pmf = sim.simulate_bivariate(repeat_degree_counts)  # {(U,V): p}

    # Simulated CDF from simulated PMF
    def jaccard_cdf_sim(t: float) -> float:
        s = 0.0
        for (u, v), p in pmf.items():
            # u>0 in practice; include v==0 (J=0) and v/u < t
            if v == 0 or (u > 0 and v / u < t):
                s += p
        return s

    # Combinatorial CDF using combinatorial result
    comb_biv = CombinatorialBivariate(total_number, shard_sizes)
    jaccard_comb = CombinatorialJaccard(total_number, shard_sizes, comb_biv)

    for t in thresholds:
        jaccard_cdf_sim_t = jaccard_cdf_sim(t)
        jaccard_cdf_comb_t = jaccard_comb.jaccard_cdf_analytical(t)
        assert jaccard_cdf_comb_t == pytest.approx(jaccard_cdf_sim_t, abs=0.02)

    # Compare the mean of the simulated and combinatorial Jaccard indices
    jaccard_mu_sim = sum(v / u * p for (u, v), p in pmf.items())
    jaccard_mu_comb = jaccard_comb.jaccard_mu()
    assert jaccard_mu_comb == pytest.approx(jaccard_mu_sim, abs=0.02)
