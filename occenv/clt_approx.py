"""
Approximated results for the univariate and bivariate distributions using CLT.
"""

import numpy as np
import itertools
from math import prod
from typing import Iterable


class CltApproxResult:
    """
    Approximated results for the union, intersection, bivariate, and jaccard index distributions using CLT.
    """

    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)
        self.alpha = np.array(shard_sizes) / self.total_number

    def mu_sum_indicators(self, probability: float) -> float:
        """
        X = sum of indicator random variables, each with probability of getting selected as p.
        E[X] = sum of E[I_i] = sum of p = total_number * p
        """
        return self.total_number * probability

    def var_individuals(self, probability: float) -> float:
        """
        Var(X) = sum of Var(I_i) = sum of p * (1 - p) = total_number * p * (1 - p)
        """
        return self.total_number * probability * (1 - probability)

    # --- Set Union approximated results ---

    def union_p_approx(self) -> float:
        return 1 - np.prod(1 - self.alpha)

    def union_var_approx(self) -> float:
        """
        Variance of individuals + covariance
        """
        p_union = self.union_p_approx()
        return self.var_individuals(p_union) + self.total_number * (
            self.total_number - 1
        ) * (
            -np.prod(1 - self.alpha) ** 2
            + np.prod(
                [
                    (self.total_number - shard)
                    * (self.total_number - shard - 1)
                    / (self.total_number * (self.total_number - 1))
                    for shard in self.shard_sizes
                ]
            )
        )

    def union_sd_approx(self) -> float:
        return np.sqrt(self.union_var_approx())

    def sigma_value(
        self,
    ):  # N * sigma value is the expected total union, using inclusion-exclusion principle
        sigma = 0.0
        for k in range(
            1, self.party_number + 1
        ):  # the outer summation - loop over k from 1 to m
            sum_k = 0.0
            for combo in itertools.combinations(
                range(self.party_number), k
            ):  # the inner summation - loop over all combinations of m choose k
                sum_k += self.rho(combo)
            sigma += ((-1) ** (k + 1)) * sum_k
        return sigma

    # --- Set Intersect approximated results ---

    def intersection_p_approx(self) -> float:
        return np.prod(self.alpha)

    def intersection_var_approx(self) -> float:
        """
        Variance of individuals + covariance
        """
        p_intersect = self.intersection_p_approx()
        return self.var_individuals(p_intersect) + self.total_number * (
            self.total_number - 1
        ) * (
            np.prod(
                self.alpha**2 + (self.alpha**2 - self.alpha) / (self.total_number - 1)
            )
            - np.prod(self.alpha) ** 2
        )

    def intersection_sd_approx(self) -> float:
        return np.sqrt(self.intersection_var_approx())

    def rho(self, indices: Iterable[int]) -> float:
        product = prod(self.shard_sizes[i] for i in indices)
        k = len(indices)
        result = product / (self.total_number**k)
        return result

    def occ_value(self):  # N * OCC value is the expected total intersection
        return self.rho(list(range(self.party_number)))

    # --- Degree distribution (general case) ---

    def degree_pmf(self) -> np.ndarray:
        """
        p_d(alpha) for d=0..m where p_d is the coefficient of t^d in
        prod_r ((1-alpha_r)+alpha_r t).
        """
        coeffs = np.array([1.0])
        for a in self.alpha:
            coeffs = np.convolve(coeffs, [1.0 - a, a])
        return coeffs

    def degree_prob(self, d: int) -> float:
        return float(self.degree_pmf()[d])

    def z_mean(self, d: int) -> float:
        """
        E[Z_d] = N * p_d(alpha).
        """
        return self.total_number * self.degree_prob(d)

    def _pair_shard_probs(self) -> list[tuple[float, float, float, float]]:
        """
        Per-shard probabilities for two distinct items (u,v):
        (p00, p10, p01, p11) where p10=P(u in shard, v not), etc.
        """
        n = self.total_number
        if n < 2:
            return [(1.0, 0.0, 0.0, 0.0) for _ in self.shard_sizes]
        out = []
        for n_r in self.shard_sizes:
            p11 = (n_r / n) * ((n_r - 1) / (n - 1))
            p10 = (n_r / n) * ((n - n_r) / (n - 1))
            p01 = p10
            p00 = ((n - n_r) / n) * ((n - n_r - 1) / (n - 1))
            out.append((p00, p10, p01, p11))
        return out

    def pair_degree_pmf(self) -> np.ndarray:
        """
        q_{a,b} for a,b=0..m where q_{a,b}=P(R(u)=a, R(v)=b) for u!=v.
        """
        m = self.party_number
        coeffs = np.zeros((1, 1), float)
        coeffs[0, 0] = 1.0
        for p00, p10, p01, p11 in self._pair_shard_probs():
            kernel = np.array([[p00, p01], [p10, p11]], float)
            new = np.zeros((coeffs.shape[0] + 1, coeffs.shape[1] + 1), float)
            for i in range(coeffs.shape[0]):
                for j in range(coeffs.shape[1]):
                    new[i : i + 2, j : j + 2] += coeffs[i, j] * kernel
            coeffs = new
        if coeffs.shape != (m + 1, m + 1):
            coeffs = coeffs[: m + 1, : m + 1]
        return coeffs

    def pair_degree_prob(self, a: int, b: int) -> float:
        return float(self.pair_degree_pmf()[a, b])

    def z_var(self, a: int) -> float:
        """
        Var(Z_a) = N p_a(1-p_a) + N(N-1)(q_{a,a} - p_a^2).
        """
        n = self.total_number
        p_a = self.degree_prob(a)
        if n < 2:
            return n * p_a * (1.0 - p_a)
        q_aa = self.pair_degree_prob(a, a)
        return n * p_a * (1.0 - p_a) + n * (n - 1) * (q_aa - p_a * p_a)

    def z_cov(self, a: int, b: int) -> float:
        """
        Cov(Z_a, Z_b) = N(N-1) q_{a,b} - N^2 p_a p_b for a!=b.
        """
        n = self.total_number
        p_a = self.degree_prob(a)
        p_b = self.degree_prob(b)
        if a == b:
            return self.z_var(a)
        if n < 2:
            return -n * p_a * p_b
        q_ab = self.pair_degree_prob(a, b)
        return n * (n - 1) * q_ab - n * n * p_a * p_b

    # --- Bivariate approximated results ---
    def bivariate_mu_approx(self) -> np.ndarray:
        """
        Output the mean matrix.
        """
        return np.array(
            [
                self.mu_sum_indicators(self.union_p_approx()),
                self.mu_sum_indicators(self.intersection_p_approx()),
            ]
        )

    def bivariate_cov_approx(self) -> float:
        """
        Covariance between the union and intersection distributions.
        """
        a = self.total_number * self.intersection_p_approx()
        b = (
            self.total_number
            * (self.total_number - 1)
            * self.intersection_p_approx()
            * (
                1
                - np.prod(
                    [
                        (self.total_number - n_i) / (self.total_number - 1)
                        for n_i in self.shard_sizes
                    ]
                )
            )
        )
        c = self.total_number**2 * self.intersection_p_approx() * self.union_p_approx()
        return a + b - c

    def bivariate_matrix_approx(self) -> np.ndarray:
        """
        Output the covariance matrix.
        """
        return np.array(
            [
                [self.union_var_approx(), self.bivariate_cov_approx()],
                [self.bivariate_cov_approx(), self.intersection_var_approx()],
            ]
        )

    def bivariate_corr_approx(self) -> float:
        """
        Correlation between the union and intersection distributions, which is calculated as the covariance divided by the product of the standard deviations of the union and intersection distributions.
        """
        return self.bivariate_cov_approx() / (
            np.sqrt(self.union_var_approx()) * np.sqrt(self.intersection_var_approx())
        )

    # --- Jaccard index approximated results ---

    def jaccard_mu_approx(self) -> float:
        j = (
            self.intersection_p_approx() / self.union_p_approx()
            if self.union_p_approx() > 0
            else 0
        )
        second_delta = (
            j * self.union_var_approx() - self.bivariate_cov_approx()
        ) / self.mu_sum_indicators(self.union_p_approx()) ** 2
        return j + second_delta

    def jaccard_var_approx(self) -> float:
        j = (
            self.intersection_p_approx() / self.union_p_approx()
            if self.union_p_approx() > 0
            else 0
        )
        a = (
            self.intersection_var_approx()
            + j**2 * self.union_var_approx()
            - 2 * j * self.bivariate_cov_approx()
        )
        b = self.mu_sum_indicators(self.union_p_approx()) ** 2
        return a / b

    def jaccard_mu_approx_simplified(self) -> float:
        return (
            self.intersection_p_approx() / self.union_p_approx()
            if self.union_p_approx() > 0
            else 0
        )


if __name__ == "__main__":
    ar = CltApproxResult(2000, [1800, 1900, 1800, 1800])
    # --- Univariate approximated results ---
    print("probability (intersection):", ar.intersection_p_approx())
    print("probability (union):", ar.union_p_approx())
    print("mean (union):", ar.mu_sum_indicators(ar.union_p_approx()))
    print("variance (union):", ar.union_var_approx())
    print("variance (iid union):", ar.var_individuals(ar.union_p_approx()))
    print("mean (intersection):", ar.mu_sum_indicators(ar.intersection_p_approx()))
    print("variance (intersection):", ar.intersection_var_approx())
    print(
        "variance (iid intersection):", ar.var_individuals(ar.intersection_p_approx())
    )

    # --- Bivariate approximated results ---
    print("covariance (bivariate):", ar.bivariate_cov_approx())
    print("correlation (bivariate):", ar.bivariate_corr_approx())
    print("mean (bivariate):", ar.bivariate_mu_approx())
    print("matrix (bivariate):", ar.bivariate_matrix_approx())
    print("mean (jaccard):", ar.jaccard_mu_approx())
    print("variance (jaccard):", ar.jaccard_var_approx())
    print("mean (jaccard simplified):", ar.jaccard_mu_approx_simplified())
