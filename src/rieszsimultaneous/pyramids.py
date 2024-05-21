from typing import Literal, Sequence, Tuple, Union
import numpy as np

from rieszsimultaneous.algorithms.riesz_transform import compute_approximate_riesz_pyramid, compute_ideal_riesz_pyramid
from rieszsimultaneous.utils.processing import max_pyramid_size


class RieszPyramid:
    def __init__(
        self, 
        frame: np.ndarray, 
        n_levels: Union[int, None] = None,
        algorithm: Union[Literal["approximate", "ideal"], callable] = "approximate"
    ):
        """Initialize Riesz pyramid from frame

        Parameters
        ----------
        frame : np.ndarray
            2-D or 3-D numpy array with dimensions (H, W) or (H, W, C)
        n_levels : int
            Number of levels in the pyramid, without considering the residue
        """
        self._set_frame(frame)
        self._set_algorithm(algorithm)
        self._set_n_levels(n_levels)

        self._compute_pyramid()

    def as_list_of_numpy(self) -> Tuple[Sequence[np.ndarray], Sequence[np.ndarray], Sequence[np.ndarray]]:
        return self._laplacian, self._riesz_x, self._riesz_y
        
    def _compute_pyramid(self):
        if callable(self._algorithm):
            self._laplacian, self._riesz_x, self._riesz_y = self._algorithm(
                self._frame, self._n_levels
            )
        elif self._algorithm == "approximate":
            self._laplacian, self._riesz_x, self._riesz_y = compute_approximate_riesz_pyramid(
                self._frame, self._n_levels
            )
        elif self._algorithm == "ideal":
            self._laplacian, self._riesz_x, self._riesz_y = compute_ideal_riesz_pyramid(
                self._frame, self._n_levels
            )
        else:
            raise ValueError(f"Invalid algorithm set: {self._algorithm}")

    def _set_n_levels(self, n_levels: Union[int, None]):
        if n_levels is None:
            self._n_levels = max_pyramid_size(self._frame_height, self._frame_width) - 1
        else:
            self._n_levels = n_levels

    def _set_frame(self, frame: np.ndarray):
        self._frame = frame

    def _set_algorithm(self, algorithm):
        self._algorithm = algorithm
    
    @property
    def _frame_height(self):
        return self._frame.shape[0]
    
    @property
    def _frame_width(self):
        return self._frame.shape[1]
