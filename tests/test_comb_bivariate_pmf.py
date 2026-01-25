"""
Test that the simulated bivariate event probability matches the combinatorial bivariate probability.
"""

import pytest
from occenv.simulate import Simulate
from occenv.comb_bivariate import CombinatorialBivariate


@pytest.mark.parametrize(
    "shard_sizes, target_uv",
    [
        ((30, 35), (50, 20)),
        ((60, 10), (60, 10)),
        ((70, 20), (70, 15)),
        ((50, 30), (60, 10)),
    ],
)
def test_bivariate_event_probability(shard_sizes, target_uv):
    """
    Test that the simulated bivariate event (|U|=u and |V|=v) probability
    matches the combinatorial bivariate event probability.
    """
    total_number = 100
    repeats = int(1e6)

    sim = Simulate(total_number, shard_sizes)
    degree_counts = sim.simulate_degree_count_repeat(repeat=repeats)
    pmf = sim.simulate_bivariate(degree_counts)

    # Simulated results
    p_sim = pmf.get(target_uv, 0.0)

    # Combinatorial results
    p_comb = CombinatorialBivariate(total_number, shard_sizes).bivariate_prob(
        target_uv[0], target_uv[1]
    )

    assert p_comb == pytest.approx(p_sim, abs=0.01)
