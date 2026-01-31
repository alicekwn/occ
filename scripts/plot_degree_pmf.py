"""
Plot the PMF of
(1) Z_d (number of items with degree d) and overlay simulation.
(2) A_D=sum_d in D Z_d (number of items with degrees in D) and overlay simulation.
"""

import numpy as np
import matplotlib.pyplot as plt

from occenv.clt_degree_approx import CltDegreeVector
from occenv.simulate import Simulate
from occenv.utils import norm_pdf


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

    # Normal approximation: Z_d ~ Normal(mean=z_mean, var=z_var)
    mu = approx.z_mean(d)
    var = approx.z_var(d)
    sigma = np.sqrt(var) if var > 0 else 0.0
    xs = np.arange(total_number + 1)
    pdf_norm = norm_pdf(xs, mu, sigma)

    # Simulation: counts of Z_d across repeats
    degree_counts = sim.simulate_degree_count_repeat(repeats)
    z_d_counts = degree_counts[:, d]
    pmf_sim = np.bincount(z_d_counts, minlength=total_number + 1) / repeats

    # Plot
    plt.figure()
    plt.bar(xs, pmf_sim, alpha=0.5, label=f"Simulation (repeats={repeats})")
    plt.plot(xs, pdf_norm, "r-", lw=2, label=f"Normal (μ={mu:.2f}, σ={sigma:.2f})")
    plt.title(f"PMF of Z_d for d={d} (N={total_number}, sizes={shard_sizes})")
    plt.xlabel("Z_d")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_degrees_pmf(
    total_number: int,
    shard_sizes: tuple[int, ...],
    degrees: list[int],
    repeats: int = 1000,
):
    """
    Plot the PMF of Z_D (number of items with degrees in D) and overlay simulation.
    """

    # Approximated results: Z_D ~ Normal(mean=z_means, var=z_vars)
    approx = CltDegreeVector(total_number, shard_sizes)
    approx_mean = approx.z_means(degrees)
    approx_var = approx.z_vars(degrees)
    approx_sigma = np.sqrt(approx_var) if approx_var > 0 else 0.0
    xs = np.arange(total_number + 1)
    pdf_norm = norm_pdf(xs, approx_mean, approx_sigma)

    # Simulation: counts of Z_D across repeats
    sim = Simulate(total_number, shard_sizes)
    degree_counts = sim.simulate_degree_count_repeat(repeats)
    z_d_counts = degree_counts[:, degrees].sum(axis=1)
    pmf_sim = np.bincount(z_d_counts, minlength=total_number + 1) / repeats

    plt.figure()
    plt.bar(xs, pmf_sim, alpha=0.5, label=f"Simulation (repeats={repeats})")
    plt.plot(
        xs,
        pdf_norm,
        "r-",
        lw=2,
        label=f"Normal (μ={approx_mean:.2f}, σ={approx_sigma:.2f})",
    )
    plt.title(f"PMF of Z_D for D={degrees} (N={total_number}, sizes={shard_sizes})")
    plt.xlabel("Z_D")
    plt.ylabel("Probability")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_degree_pmf_weighted(
    total_number: int,
    shard_sizes: tuple[int, ...],
    weights: list[float],  # len(weights) = len(shard_sizes)+1
    repeats: int = 1000,
):
    """
    Plot the PDF of T=LZ, where Z is degree-count vector and L is a vector of weights.
    On top of the simulation PMF (discrete bin widths).
    """

    # Simulation: counts of T across repeats
    sim = Simulate(total_number, shard_sizes)
    degree_counts = sim.simulate_degree_count_repeat(repeats)
    t_counts = degree_counts.dot(weights)
    t_min, t_max = t_counts.min(), t_counts.max()

    # Check if all weights are integers (then t_counts will be integers too)
    all_int_weights = all(float(w).is_integer() for w in weights)
    if all_int_weights:
        # Use bincount for integer weights (exact PMF)
        t_shifted = (t_counts - t_min).astype(int)
        pmf_sim = np.bincount(t_shifted) / repeats
        xs = np.arange(len(pmf_sim)) + t_min
        bar_width = 1.0
    else:
        # Use histogram for float weights
        # Determine bin width based on the smallest weight fraction
        bar_width = 0.5  # reasonable default for half-integer weights
        bins = np.arange(t_min - bar_width / 2, t_max + bar_width, bar_width)
        counts, edges = np.histogram(t_counts, bins=bins)
        pmf_sim = counts / repeats
        xs = (edges[:-1] + edges[1:]) / 2  # bin centers

    # Approximated results: T=LZ ~ Normal(mean=approx_mean, var=approx_var)
    approx = CltDegreeVector(total_number, shard_sizes)
    approx_mean = approx.z_means_weighted(weights)
    approx_var = approx.z_vars_weighted(weights)
    approx_sigma = np.sqrt(approx_var) if approx_var > 0 else 0.0
    pdf_norm = norm_pdf(xs, approx_mean, approx_sigma)

    # Create figure with twin axes
    _, ax1 = plt.subplots()

    # Left axis: PDF (continuous normal distribution)
    ax1.plot(
        xs,
        pdf_norm,
        "r-",
        lw=2,
        label=f"Normal PDF (μ={approx_mean:.2f}, σ={approx_sigma:.2f})",
    )
    ax1.set_xlabel("T")
    ax1.set_ylabel("PDF (Probability Density)", color="r")
    ax1.tick_params(axis="y", labelcolor="r")
    ax1.set_ylim(bottom=0)

    # Right axis: PMF (discrete probabilities)
    ax2 = ax1.twinx()
    ax2.bar(
        xs,
        pmf_sim,
        width=bar_width,
        alpha=0.5,
        label=f"Simulation PMF (repeats={repeats})",
        color="blue",
    )
    ax2.set_ylabel("PMF (Probability Mass)", color="b")
    ax2.tick_params(axis="y", labelcolor="b")

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    plt.title(f"PMF of T=LZ for L={weights} (N={total_number}, sizes={shard_sizes})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # plot_degree_pmf(
    #     total_number=100,
    #     shard_sizes=(
    #         20,
    #         25,
    #         30,
    #         20,
    #     ),
    #     d=2,
    #     repeats=int(1e5),
    # )
    # plot_degrees_pmf(
    #     total_number=100,
    #     shard_sizes=(20, 30, 40, 50),
    #     degrees=[2, 3, 4],
    #     repeats=int(1e5),
    # )
    plot_degree_pmf_weighted(
        total_number=100,
        shard_sizes=(20, 30, 40, 50),
        weights=[-2, 3, 2.1, 4.2, -1.5],
        repeats=int(1e5),
    )
