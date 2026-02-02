"""
This script plots the combinatorial results (PMF) of the union and the intersection of the parties,
together with the CLT approximated normal distribution.

The script also plots the reverse cumulative distribution and the cumulative distribution.
"""

import numpy as np
from occenv.comb_univariate import CombinatorialUnivariate
from occenv.utils import (
    mu_calculation,
    sd_calculation,
    mae_calculation,
    sse_calculation,
    norm_pdf,
)
from occenv.plotting_2d import plot_pmf_with_normals
from occenv.clt_special_case import CltSpecialCase

N = 100
shard_sizes = (10, 10, 10, 10)
m = len(shard_sizes)

comb = CombinatorialUnivariate(N, shard_sizes)
clt_approx = CltSpecialCase(N, shard_sizes)

numbers_range = np.arange(0, N + 1, 1)
x_continuous = np.linspace(min(numbers_range) - 1, max(numbers_range) + 1, 500)

problems = {
    # "Union": "u",
    "Intersection": "v"
}

for problem, label in problems.items():
    # Calculate the PMF for the problem
    pmf = []
    prob_method = getattr(comb, f"{problem.lower()}_prob")
    for number_in_range in numbers_range:
        probability = prob_method(number_in_range)
        pmf.append(probability)

    # Calculate the combinatorial mean and standard deviation of the discrete PMF
    mu_comb = mu_calculation(numbers_range, pmf)
    sigma_comb = sd_calculation(numbers_range, pmf)
    normal_comb = norm_pdf(x_continuous, mu_comb, sigma_comb)

    # Calculate the approximated mean and standard deviation of the discrete PMF (using CLT)
    p_approx = getattr(clt_approx, f"{problem.lower()}_p_approx")
    sd_approx_method = getattr(clt_approx, f"{problem.lower()}_sd_approx")
    mu_approx = clt_approx.mu_sum_indicators(p_approx())
    sd_approx = sd_approx_method()
    normal_approx = norm_pdf(x_continuous, mu_approx, sd_approx)

    # Error between the two normal distributions
    mae_norm = mae_calculation(normal_comb, normal_approx)
    sse_norm = sse_calculation(normal_comb, normal_approx)

    # Assume iid, calculate the variance
    sd_assume_iid = np.sqrt(clt_approx.var_individuals(p_approx()))

    # Error between the approximated variance and the variance assuming iid
    mae_var = mae_calculation(sigma_comb, sd_assume_iid)
    sse_var = sse_calculation(sigma_comb, sd_assume_iid)

    # Plot the bar chart of the discrete PMF and plot the normal approximation on top of the PMF
    plot_pmf_with_normals(
        numbers_range,
        pmf,
        x_continuous,
        mu_comb,
        sigma_comb,
        mu_approx,
        sd_approx,
        xlabel=f"{label}",
        title=f"PMF for N={N},$S_{m}$={shard_sizes}",
    )

    # Compare the approximated mean and standard deviation with the combinatorial mean and standard deviation
    print(
        f"Combinatorial result mean for {problem} (μ): {mu_comb} \nCLT approximated result mean for {problem} (μ): {mu_approx}"
    )
    print(
        f"Combinatorial result standard deviation for {problem} (σ): {sigma_comb} \nCLT approximated result standard deviation for {problem} (σ): {sd_approx}"
    )
    print(f"Error between the two normals: MAE = {mae_norm:.5f}, SSE = {sse_norm:.5f}")

    # To show that the iid assumption is not always correct, we plot the error between the approximated variance and the variance assuming iid
    plot_pmf_with_normals(
        numbers_range,
        pmf,
        x_continuous,
        mu_comb,
        sigma_comb,
        mu_approx,
        sd_assume_iid,
        xlabel=f"{label}",
        title=f"PMF for N={N},$S_{m}$={shard_sizes}, compare with iid normal approximation",
    )

    print(
        f"Error between the approximated variance and the variance assuming iid: MAE = {mae_var:.5f}, SSE = {sse_var:.5f}"
    )

    # Plot the complementary cumulative distribution function and the cumulative distribution function
    # cdf = np.cumsum(pmf)
    # ccdf = np.cumsum(pmf[::-1])[::-1]
    # plot_line_graph(
    #     numbers_range,
    #     ccdf,
    #     title=f"{problem} CCDF for N={N}, $S_{m}$={shard_sizes}",
    #     xlabel="k",
    # )
    # plot_line_graph(
    #     numbers_range,
    #     cdf,
    #     title=f"{problem} CDF for N={N}, $S_{m}$={shard_sizes}",
    #     xlabel="k",
    # )
