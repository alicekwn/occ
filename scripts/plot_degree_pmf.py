"""
Plot the PMF of Z_d (number of items with degree d) and overlay simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from occenv.clt_degree_approx import CltDegreeVector
from occenv.simulate import Simulate
from occenv.utils import discretize_normal_pmf


def plot_degree_pmf(
    total_number: int,
    shard_sizes: tuple[int, ...],
    d: int = 2,
    repeats: int = 5000,
):
    """
    Plot the PMF of Z_d (number of items with degree d) and overlay simulation.
    """
    approx = CltDegreeVector(total_number, shard_sizes)
    sim = Simulate(total_number, shard_sizes)

    # Simulation: counts of Z_d across repeats
    degree_counts = sim.simulate_degree_count_repeat(repeats)
    z_d_counts = degree_counts[:, d]
    pmf_sim = np.bincount(z_d_counts, minlength=total_number + 1) / repeats

    # Normal approximation: Z_d ~ Normal(mean=z_mean, var=z_var)
    mu = approx.z_mean(d)
    var = approx.z_var(d)
    sigma = np.sqrt(var) if var > 0 else 0.0
    xs = np.arange(total_number + 1)
    pmf_norm = discretize_normal_pmf(xs, mu, sigma)

    # Plot
    plt.figure()
    plt.bar(xs, pmf_sim, alpha=0.5, label=f"Simulation (repeats={repeats})")
    plt.plot(xs, pmf_norm, "r-", lw=2, label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")
    plt.title(f"PMF of Z_d for d={d} (N={total_number}, sizes={shard_sizes})")
    plt.xlabel("Z_d")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_degree_pmf(
        total_number=100,
        shard_sizes=(
            20,
            25,
            30,
            20,
        ),
        d=2,
        repeats=int(1e5),
    )
    # plot_degree_pmf(
    #     total_number=100, shard_sizes=(90, 60, 70, 80), d=4, repeats=int(1e5)
    # )
