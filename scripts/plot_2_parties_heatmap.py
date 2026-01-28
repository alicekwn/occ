"""
This script plots the heatmap of results when there are 2 parties.
"""

import numpy as np
from occenv.comb_univariate import CombinatorialUnivariate
from occenv.comb_bivariate import CombinatorialBivariate
from occenv.comb_jaccard import CombinatorialJaccard
from occenv.clt_special_case import CltSpecialCase
from occenv.plotting_3d import plot_heatmap

N = 50
n_vals = np.arange(1, N)
x, y = np.meshgrid(n_vals, n_vals)

plot_dict = {
    "collusion_prob": {
        "title": f"Full coverage probability when $N={N},m=2$",
        "z": np.vectorize(
            lambda n1, n2: CombinatorialUnivariate(N, (n1, n2)).union_prob(N)
        )(x, y),
    },
    "sigma": {
        "title": f"$\sigma$ with $N={N},m=2$",
        "z": np.vectorize(lambda n1, n2: CltSpecialCase(N, (n1, n2)).sigma_value())(
            x, y
        ),
    },
    "occ": {
        "title": f"OCC with $N={N},m=2$",
        "z": np.vectorize(lambda n1, n2: CltSpecialCase(N, (n1, n2)).occ_value())(x, y),
    },
    "expected_jaccard": {
        "title": f"Expected Jaccard index with $N={N}$",
        "z": np.vectorize(
            lambda n1, n2: CombinatorialJaccard(
                N, (n1, n2), CombinatorialBivariate(N, (n1, n2))
            ).jaccard_mu()
        )(x, y),
    },
    "estimated_jaccard": {
        "title": f"Estimated Jaccard index with $N={N}$",
        "z": np.vectorize(
            lambda n1, n2: CltSpecialCase(N, (n1, n2)).jaccard_mu_approx_simplified()
        )(x, y),
    },
    "jaccard_difference": {
        "title": f"Difference between Expected and Estimated Jaccard index with $N={N}$",
        "z": np.vectorize(
            lambda n1, n2: CombinatorialJaccard(
                N, (n1, n2), CombinatorialBivariate(N, (n1, n2))
            ).jaccard_mu()
        )(x, y)
        - np.vectorize(
            lambda n1, n2: CltSpecialCase(N, (n1, n2)).jaccard_mu_approx_simplified()
        )(x, y),
        "vmax": 0.05,
    },
}

for key, value in plot_dict.items():
    plot_heatmap(
        x=x,
        y=y,
        z=value["z"],
        xlabel="$n_1$",
        ylabel="$n_2$",
        title=value["title"],
        cmap="Blues",
        vmin=0,
        vmax=value.get("vmax", 1),
    )
