from occenv.comb_bivariate import CombinatorialBivariate
from occenv.plotting_3d import (
    plot_heatmap_ellipse,
    # plot_surface_3d,
    # plot_surface_plotly,
)
from occenv.utils_bivariate import Gaussian2D

N = 100
shard_sizes = (30, 30, 40)

# Build grid (U, V, Z) from the combinatorial model
comb_biv = CombinatorialBivariate(total_number=N, shard_sizes=shard_sizes)
U_vals, V_vals, Z_vals = comb_biv.bivariate_grid()
mu = comb_biv.bivariate_mu()
Sigma = comb_biv.bivariate_matrix()


# Plot 2D heatmap + ellipses overlays indicating confidence levels
plot_heatmap_ellipse(
    U_vals,
    V_vals,
    Z_vals,
    mu,
    Sigma,
    color_map="hot_r",
    title=f"Bivariate distribution for N={N}, sizes={shard_sizes}",
    outpath=None,
)

## Plot 3D surface
# plot_surface_3d(
#     U_vals,
#     V_vals,
#     Z_vals,
#     title=f"Bivariate distribution for N={N}, sizes={shard_sizes} — 3D",
# )

## Plot 3D surface (interactive)
# plot_surface_plotly(
#     U_vals,
#     V_vals,
#     Z_vals,
#     title=f"Bivariate distribution for N={N}, sizes={shard_sizes} — 3D (interactive)",
# )

# Print mean vector, covariance matrix, eigenvalues and eigenvectors
print("mu =", mu)
print("Sigma =\n", Sigma)
print("Eigenvalues =", Gaussian2D(mu, Sigma).evals)
print("Eigenvectors (columns) =\n", Gaussian2D(mu, Sigma).evecs)
