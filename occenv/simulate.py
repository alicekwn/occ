from joblib import Parallel, delayed
from occenv.env import DataShare
from collections import Counter
import numpy as np


class Simulate:
    def __init__(self, total_number: int, shard_sizes: list[int]):
        self.total_number = total_number
        self.shard_sizes = shard_sizes
        self.m = len(shard_sizes)

    def _simulate_level_count_once(self) -> list[int]:
        """
        Single run → level counts of the data share.
        """
        new_mpc = DataShare(self.total_number)
        shards = [set(new_mpc.create_shard(s)) for s in self.shard_sizes]
        level_counts = [0] * (self.m + 1)
        for i in range(self.total_number):
            count = 0
            for shard in shards:
                if i in shard:
                    count += 1
            level_counts[count] += 1
        return level_counts

    def simulate_level_count_repeat(self, repeat: int) -> list[int]:
        """
        Repeat the simulation of level counts and return the average level counts for each level.
        """
        levels = np.array(
            Parallel(n_jobs=-1)(
                delayed(self._simulate_level_count_once)() for _ in range(repeat)
            )
        )  # the output is a list of lists, each list is the level counts of a single run
        level_average = [float(round(sum(x) / repeat, 4)) for x in zip(*levels)]
        return level_average

    def _simulate_bivariate_once(self) -> tuple[int, int]:
        """
        Single run → (U, V) for ALL shards:
        U = union size, V = intersection size.
        """
        if self.m == 0:
            return (0, 0)

        new_mpc = DataShare(self.total_number)
        shards = [set(new_mpc.create_shard(s)) for s in self.shard_sizes]
        U = len(set().union(*shards))
        V = len(set.intersection(*shards))
        return U, V

    def simulate_bivariate_repeat(
        self, repeat: int, block: int = 10_000
    ) -> dict[tuple[int, int], float]:
        """
        Bivariate PMF over (U, V): {(U, V): probability}.
        Runs in blocks to avoid storing all raw samples.
        """

        def _run_block(block_number: int) -> Counter:
            c = Counter()
            for _ in range(block_number):
                c[self._simulate_bivariate_once()] += 1
            return c

        q, r = divmod(repeat, block)
        blocks = [block] * q + ([r] if r else [])
        parts = Parallel(n_jobs=-1)(
            delayed(_run_block)(block_number) for block_number in blocks
        )

        total = Counter()
        for part in parts:
            total.update(part)

        return {uv: cnt / repeat for uv, cnt in total.items()}

    def _simulate_union_once(self) -> int:
        U, _ = self._simulate_bivariate_once()
        return U

    def simulate_union_repeat(self, repeat: int) -> list[int]:
        return Parallel(n_jobs=-1)(
            delayed(self._simulate_union_once)() for _ in range(repeat)
        )

    def _simulate_intersection_once(self) -> int:
        _, V = self._simulate_bivariate_once()
        return V

    def simulate_intersection_repeat(self, repeat: int) -> list[int]:
        return Parallel(n_jobs=-1)(
            delayed(self._simulate_intersection_once)() for _ in range(repeat)
        )


if __name__ == "__main__":
    N = 100
    n = [50, 63, 75]

    simulator = Simulate(N, n)
    repeats = int(1e6)

    # union = simulator.simulate_union_repeat(repeats)
    # print("The average union is ", sum(union) / repeats)

    # intersection = simulator.simulate_intersection_repeat(repeats)
    # print("The average intersection is ", sum(intersection) / repeats)

    repeat_level_counts = simulator.simulate_level_count_repeat(repeats)
    print("The average level counts are ", repeat_level_counts)
