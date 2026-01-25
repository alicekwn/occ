"""
Test that the simulated univariate probabilities match the combinatorial univariate probabilities.
"""

from collections import Counter
import pytest
from occenv.simulate import Simulate
from occenv.comb_univariate import CombinatorialUnivariate
from occenv.utils import mu_calculation, var_calculation


@pytest.mark.parametrize(
    "shard_sizes",
    [
        # (7, 9),
        # (7),
        # (10),
        # (6, 3),
        (5, 6),
        (10, 6),
        (3, 2, 4),
        (7, 6, 9),
        (3, 5, 7, 3),
    ],
)
def test_univariate_pmf_and_mean(shard_sizes):
    """
    Test that the simulated univariate (union and intersection) probabilities
    match the combinatorial univariate probabilities.
    """
    total_number = 10
    repeats = int(1e6)

    sim = Simulate(total_number, shard_sizes)
    comb = CombinatorialUnivariate(total_number, shard_sizes)

    # --- Simulated PMFs from samples ---
    degree_counts = sim.simulate_degree_count_repeat(repeat=repeats)
    union_samples = total_number - degree_counts[:, 0]
    intersection_samples = degree_counts[:, len(shard_sizes)]

    count_u = Counter(union_samples)
    count_v = Counter(intersection_samples)

    x_u = list(range(0, total_number + 1))
    x_v = list(range(0, min(shard_sizes) + 1))

    pmf_u_sim = [count_u.get(u, 0) / repeats for u in x_u]
    pmf_v_sim = [count_v.get(v, 0) / repeats for v in x_v]

    # --- Combinatorial PMFs ---
    pmf_u_comb = [comb.union_prob(u) for u in x_u]
    pmf_v_comb = [comb.intersection_prob(v) for v in x_v]

    # --- Compare PMFs element-wise ---
    for p_sim, p_comb in zip(pmf_u_sim, pmf_u_comb):
        assert p_comb == pytest.approx(p_sim, abs=0.01)
    for p_sim, p_comb in zip(pmf_v_sim, pmf_v_comb):
        assert p_comb == pytest.approx(p_sim, abs=0.01)

    # --- Sanity checks ---
    assert sum(pmf_u_sim) == pytest.approx(1.0, abs=1e-3)
    assert sum(pmf_v_sim) == pytest.approx(1.0, abs=1e-3)
    assert sum(pmf_u_comb) == pytest.approx(1.0, abs=1e-9)
    assert sum(pmf_v_comb) == pytest.approx(1.0, abs=1e-9)

    # --- Compare means ---
    mean_union_sim = mu_calculation(x_u, pmf_u_sim)
    mean_union_comb = comb.union_mu()
    mean_inter_sim = mu_calculation(x_v, pmf_v_sim)
    mean_inter_comb = comb.intersection_mu()

    assert mean_union_comb == pytest.approx(mean_union_sim, abs=0.01)
    assert mean_inter_comb == pytest.approx(mean_inter_sim, abs=0.01)

    # --- Compare variances ---
    var_union_sim = var_calculation(x_u, pmf_u_sim)
    var_union_comb = comb.union_var()
    var_inter_sim = var_calculation(x_v, pmf_v_sim)
    var_inter_comb = comb.intersection_var()
    assert var_union_comb == pytest.approx(var_union_sim, abs=0.01)
    assert var_inter_comb == pytest.approx(var_inter_sim, abs=0.01)
