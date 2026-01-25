"""
Test that the simulated univariate probabilities match the analytical univariate probabilities.
"""

from collections import Counter
import pytest
from occenv.simulate import Simulate
from occenv.analytical_univariate import AnalyticalUnivariate
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
    match the analytical univariate probabilities.
    """
    total_number = 10
    repeats = int(1e6)

    sim = Simulate(total_number, shard_sizes)
    ana = AnalyticalUnivariate(total_number, shard_sizes)

    # --- Empirical PMFs from samples ---
    degree_counts = sim.simulate_degree_count_repeat(repeat=repeats)
    union_samples = total_number - degree_counts[:, 0]
    intersection_samples = degree_counts[:, len(shard_sizes)]

    count_u = Counter(union_samples)
    count_v = Counter(intersection_samples)

    x_u = list(range(0, total_number + 1))
    x_v = list(range(0, min(shard_sizes) + 1))

    pmf_u_emp = [count_u.get(u, 0) / repeats for u in x_u]
    pmf_v_emp = [count_v.get(v, 0) / repeats for v in x_v]

    # --- Analytical PMFs ---
    pmf_u_ana = [ana.union_prob(u) for u in x_u]
    pmf_v_ana = [ana.intersection_prob(v) for v in x_v]

    # --- Compare PMFs element-wise ---
    for p_emp, p_ana in zip(pmf_u_emp, pmf_u_ana):
        assert p_ana == pytest.approx(p_emp, abs=0.01)
    for p_emp, p_ana in zip(pmf_v_emp, pmf_v_ana):
        assert p_ana == pytest.approx(p_emp, abs=0.01)

    # --- Sanity checks ---
    assert sum(pmf_u_emp) == pytest.approx(1.0, abs=1e-3)
    assert sum(pmf_v_emp) == pytest.approx(1.0, abs=1e-3)
    assert sum(pmf_u_ana) == pytest.approx(1.0, abs=1e-9)
    assert sum(pmf_v_ana) == pytest.approx(1.0, abs=1e-9)

    # --- Compare means ---
    mean_union_emp = mu_calculation(x_u, pmf_u_emp)
    mean_union_ana = ana.union_mu()
    mean_inter_emp = mu_calculation(x_v, pmf_v_emp)
    mean_inter_ana = ana.intersection_mu()

    assert mean_union_ana == pytest.approx(mean_union_emp, abs=0.01)
    assert mean_inter_ana == pytest.approx(mean_inter_emp, abs=0.01)

    # --- Compare variances ---
    var_union_emp = var_calculation(x_u, pmf_u_emp)
    var_union_ana = ana.union_var()
    var_inter_emp = var_calculation(x_v, pmf_v_emp)
    var_inter_ana = ana.intersection_var()
    assert var_union_ana == pytest.approx(var_union_emp, abs=0.01)
    assert var_inter_ana == pytest.approx(var_inter_emp, abs=0.01)
