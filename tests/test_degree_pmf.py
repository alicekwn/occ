"""
Test the degree distribution p_d(alpha) against simulation.
"""

import pytest

from occenv.clt_degree_approx import CltDegreeVector
from occenv.simulate import Simulate


@pytest.mark.parametrize(
    "total_number, shard_sizes, d_values, repeats",
    [
        (100, (40, 50, 60, 70), (2, 3, 4), 5000),
    ],
)
def test_degree_pmf_matches_simulation(total_number, shard_sizes, d_values, repeats):
    """
    Test that the (CLT) approximated degree distribution matches the simulation.
    """
    approx = CltDegreeVector(total_number, shard_sizes)
    sim = Simulate(total_number, shard_sizes)
    counts = sim.simulate_degree_count_repeat(repeats)

    # Estimate p_d from simulation: average count for degree d divided by N
    p_sim = counts.mean(axis=0) / total_number
    for d in d_values:
        assert approx.degree_prob(d) == pytest.approx(p_sim[d], abs=0.03)
