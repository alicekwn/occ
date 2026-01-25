"""
Monte Carlo simulation of the degree distribution of the data share.
"""

from collections import Counter
import numpy as np
from joblib import Parallel, delayed
from occenv.env import DataShare


class Simulate:
    """
    Simulate the shards of the data, and calculate
    (1) the whole degree distribution
    (2) the bivariate distribution of (U, V) (which can be calculated from the degree distribution)
    (3) the union and intersection sizes (which can be calculated from the degree distribution)
    """

    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.m = len(shard_sizes)

    def _simulate_degree_count_once(self) -> list[int]:
        """
        Single run → all degree counts of the data share.
        """
        new_mpc = DataShare(self.total_number)
        shards = [set(new_mpc.create_shard(s)) for s in self.shard_sizes]
        degree_counts = [0] * (self.m + 1)
        for i in range(self.total_number):
            count = 0
            for shard in shards:
                if i in shard:
                    count += 1
            degree_counts[count] += 1
        return degree_counts

    def simulate_degree_count_repeat(self, repeat: int) -> list[int]:
        """
        Repeat the simulation of all degree counts
        and return the lists of all degree counts for each run.
        """
        degrees_lists = np.array(
            Parallel(n_jobs=-1)(
                delayed(self._simulate_degree_count_once)() for _ in range(repeat)
            )
        )  # the output is a list of lists, each list is the degree counts of a single run
        return degrees_lists

    def simulate_bivariate_from_degree_counts(
        self, degree_counts_array: np.ndarray
    ) -> dict[tuple[int, int], float]:
        """
        Calculate bivariate PMF from degree counts array.
        """
        repeat = degree_counts_array.shape[0]
        bivariate_counts = Counter()

        for degree_counts in degree_counts_array:
            U = int(self.total_number - degree_counts[0])
            V = int(degree_counts[self.m])
            bivariate_counts[(U, V)] += 1

        return {uv: cnt / repeat for uv, cnt in bivariate_counts.items()}


if __name__ == "__main__":
    N = 100
    n = [50, 60, 70]

    simulator = Simulate(N, n)
    REPEATS = int(1e6)

    repeat_degree_counts = simulator.simulate_degree_count_repeat(REPEATS)

    print(
        "\n".join(
            [
                f"The average degree count for degree = {i} is {sum(repeat_degree_counts[:, i]) / REPEATS}"
                for i in range(len(n) + 1)
            ]
        )
    )

    print("The average union is ", N - sum(repeat_degree_counts[:, 0]) / REPEATS)

    print(
        "The average intersection is ",
        sum(repeat_degree_counts[:, len(n)]) / REPEATS,
    )

    bivariate_pmf = simulator.simulate_bivariate_from_degree_counts(
        repeat_degree_counts
    )
    print(bivariate_pmf)
