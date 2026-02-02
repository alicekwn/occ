"""
Test the degree distribution p_d(alpha) against simulation.
"""

import pytest

from occenv.clt_degree_approx import CltDegreeVector
from occenv.simulate import Simulate


@pytest.mark.parametrize(
    "total_number, shard_sizes, repeats",
    [
        (100, (40, 50, 60, 70), 5000),
        (1000, (400, 500, 600, 700, 800), 5000),
    ],
)
def test_degree_pmf_matches_simulation(total_number, shard_sizes, repeats):
    """
    Test that the (CLT) approximated degree distribution (all degree counts) matches the simulation.
    """
    approx = CltDegreeVector(total_number, shard_sizes)
    sim = Simulate(total_number, shard_sizes)
    counts = sim.simulate_degree_count_repeat(repeats)

    # Estimate p_d from simulation: average count for degree d divided by N
    p_sim = counts.mean(axis=0) / total_number
    for d in range(len(shard_sizes) + 1):
        assert approx.degree_prob(d) == pytest.approx(p_sim[d], abs=0.03)
