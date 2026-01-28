"""
Approximated results for all the degree distributions using CLT.
"""

import numpy as np


class CltDegreeVector:
    """
    Approximated results for all the degree distributions using CLT.
    """

    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.party_number = len(shard_sizes)
        self.alpha = np.array(shard_sizes) / self.total_number

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

    # --- Degree distribution (more than one degree) ---
