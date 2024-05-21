import scipy.signal as signal
import numpy as np


class ButterworthFullVideo:
    """Butterworth filter to processes the full video at once."""

    def __init__(self, frame_rate: float, freq_low: float, freq_high: float, order: int):
        """Create Butterworth full video filter object.

        Parameters
        ----------
        frame_rate : float
            Video frame rate.
        freq_low : float
            Low frequency of the bandpass filter.
        freq_high : float
            High frequency of the bandpass filter.
        order : int
            Order of the butterworth filter.
        """        
        self.flow = freq_low
        self.fhigh = freq_high
        self.order = order
        self.fr = frame_rate

    def filter_array(self, array: np.ndarray, temporal_dimension: int=0) -> np.ndarray:
        """Filter the input array using a Butterworth filter.

        Parameters
        ----------
        array : np.ndarray
            Input array to be filtered.
        temporal_dimension : int, optional
            Axis to consider as the temporal dimension, by default 0

        Returns
        -------
        np.ndarray
            Filtered array with same dimensions as the input.
        """        
        coefficients = signal.butter(
            self.order, [self.flow, self.fhigh], btype="bandpass", fs=self.fr)

        td = temporal_dimension
        zi = np.squeeze(signal.lfilter_zi(coefficients[0], coefficients[1]))
        arr_0 = np.take(array, 0, axis=td)
        zi_arr = []
        for k in range(len(zi)):
            zi_arr.append(arr_0 * zi[k])
        zi_arr = np.squeeze(np.array(zi_arr))

        array_filtered, zf = signal.lfilter(coefficients[0], coefficients[1], array, axis=td, zi=zi_arr)
        return array_filtered
