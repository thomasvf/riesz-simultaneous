import numpy as np


class FullVideoTimeFilter:
    def filter_array(
        self, array: np.ndarray, temporal_dimension: int = 0
    ) -> np.ndarray:
        raise NotImplementedError()
