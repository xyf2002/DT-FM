from typing import Tuple


class SimulatedDevice:
    def __init__(self):
        self.cache = set()

    @classmethod
    def make_cache_key(
        cls, step: int, gemm: str, r: int, c: int, r_size: int, c_size: int
    ) -> Tuple[int, str, int]:
        """
        Create a cache tuple.

        Args:
            step (int): The step of training.
            gemm (int): The gemm name.
            r (int): The starting row.
            c (int): The starting column.
            r_size (int): The number of rows.
            c_size (int): The number of columns.
        Returns:
            Tuple[int, str, int]: The cache tuple [step, gemm, rc].
        """

        # [r,r_size, c,c_size] merge into one int64 each use int16
        r = (r << 16) + r_size
        c = (c << 16) + c_size
        rc = (r << 32) + c

        return (step, gemm, rc)

    def put_cache(self, cache: Tuple[int, int, int, int]) -> None:
        """
        Simulate putting a cache in the device.

        Args:
            cache (Tuple[int, int, int, int]): The cache to put in the device [step, gemm, r, c]
        """
        self.cache.add(cache)

    def try_cache(self, cache: Tuple[int, int, int, int]) -> bool:
        """
        Simulate trying to get a cache from the device.

        Returns:
            Tuple[int, int, int, int]: The cache from the device [step, gemm, r, c]
        """
        return cache in self.cache

    def clear_cache(self) -> None:
        """
        Simulate clearing the cache in the device.
        """
        self.cache = set()
