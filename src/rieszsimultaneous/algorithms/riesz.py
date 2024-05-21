from pathlib import Path
from typing import Callable, Literal, Sequence, Union
import numpy as np
import cv2

from rieszsimultaneous.algorithms.utils import compute_q_conj_prod, safe_divide
from rieszsimultaneous.pyramids import RieszPyramid
from rieszsimultaneous.utils.processing import (
    amplitude_weighted_blur,
    get_gaussian_pyramid,
    max_pyramid_size,
)
from rieszsimultaneous.filters import FullVideoTimeFilter


class RieszMagnificationFullVideo:
    def __init__(
        self,
        num_levels: int = None,
        alpha: float = 50,
        temporal_filter_motion: FullVideoTimeFilter = None,
        temporal_filter_amplitude: FullVideoTimeFilter = None,
        temporal_filter_residue: FullVideoTimeFilter = None,
        alpha_amplitude: float = None,
        alpha_residue: float = None,
        func_alpha: Callable = None,
        kernel_sigma: float = 2,
        ideal_riesz_transform: bool = False,
        chr_attenuation_motion: float = 0.0,
        chr_attenuation_amplitude: float = 0.0,
        chr_attenuation_residue: float = 1.0,
        pixel_mask: np.ndarray = None,
    ):
        """Initialize object for applying the Riesz-based motion and amplitude magnification

        Parameters
        ----------
        num_levels : int, optional
            Number of levels in the Riesz pyramid, including the residue. If None, 
            uses maximum size, by default None
        alpha : float, optional
            Magnification factor applied to motion, by default 50
        temporal_filter_motion : FullVideoTimeFilter, optional
            Temporal filter for the phases of the monogenic signal, by default None
        temporal_filter_amplitude : FullVideoTimeFilter, optional
            Temporal filter for the amplitudes of the monogenic signal, by default None
        temporal_filter_residue : FullVideoTimeFilter, optional
            Temporal filter for the (low-pass) residue of the Riesz pyramid, by default None
        alpha_amplitude : float, optional
            Magnification factor applied to the amplitudes. If None, uses the same as `alpha`,
            by default None
        alpha_residue : float, optional
            Magnification factor applied to the residue. If None, uses the same as `alpha`,
            by default None
        func_alpha : Callable, optional
            Function returning the value of alpha for the amplitudes as function of the 
            pyramid level. If None, uses the same value always, by default None
        kernel_sigma : float, optional
            Sigma for the kernel used in the amplitude-weighted blur, by default 2
        ideal_riesz_transform : bool, optional
            Whether to use the ideal (in frequency) transform for computing the Riesz
            pyramids, by default False
        chr_attenuation_motion : float, optional
            Chrominance attenuation factor applied to motion magnification, by default 0. 
            A value of 0 means that the chrominance channels are not magnified
        chr_attenuation_amplitude : float, optional
            Chrominance attenuation factor applied to color-changes magnification, 
            by default 0. A value of 0 means that the chrominance channels are not 
            magnified
        chr_attenuation_residue : float, optional
            Chrominance attenuation factor applied to the residue, by default 1.0
        pixel_mask : np.ndarray, optional
            ndarray with same dimensions as the input frame. The mask is decomposed in a 
            gaussian pyramid and it is used at each level in order to select the region to be
            magnified. Values should be in the range from 0 to 1.
        """
        self.num_levels = num_levels
        if self.num_levels is not None:
            self.num_levels -= (
                1  # self.num_levels doesn't include the residue, while num_levels does
            )
        self.alpha = alpha
        self.alpha_color = alpha_amplitude if alpha_amplitude is not None else alpha
        self.alpha_residue = alpha_residue if alpha_residue is not None else alpha
        self.func_alpha = func_alpha
        self.temporal_filter_motion = temporal_filter_motion
        self.temporal_filter_amplitude = temporal_filter_amplitude
        self.temporal_filter_residue = temporal_filter_residue
        self.ideal_riesz_transform = ideal_riesz_transform
        self.chr_att_motion = chr_attenuation_motion
        self.chr_att_amplitude = chr_attenuation_amplitude
        self.chr_att_residue = chr_attenuation_residue
        self.pixel_mask = pixel_mask

        self.mask_pyramid = None
        self.count = 0
        self.frames_yiq = []
        # those are like [level][time, line, col]
        self.unwrapped_phase_i = None
        self.unrapped_phase_j = None

        # these values are not modified with the temporal filter
        # so they don't need to be computed again if the filter changes
        self.unwrapped_phase_i_original = None
        self.unwrapped_phase_j_original = None
        self.residues_original = None

        # approximate amplitude for the amplituded-weighted filtering is stored here
        self.diffs_amplitude = None

        # amplitudes used for color-change magnification
        self.amplitudes = None

        self.residues = None
        self.ker_size = 5
        self.ker_sigma = kernel_sigma
        self.min_val = 1e-9
        self.rtol = 1e-5

    def append_frame(self, frame: np.ndarray):
        """Append a frame to the internally stored video.

        Frames are expected to be in the range 0 to 255.

        Parameters
        ----------
        frame : np.ndarray
            Frame to append.
        """
        self._set_num_levels_in_pyramid_if_not_set(frame)
        frame_yiq = self._rgb_to_32float_yuv(frame)
        self.frames_yiq.append(frame_yiq)
        self._set_masks_if_not_set()

    @property
    def n_frames(self):
        return len(self.frames_yiq)

    def update_pixel_mask(self, pixel_mask: np.ndarray):
        """Update the mask used for selecting the magnification region.

        Parameters
        ----------
        pixel_mask : np.ndarray, optional
            ndarray with same dimensions as the input frame. The mask is decomposed in a 
            gaussian pyramid and it is used at each level in order to select the region to be
            magnified. Values should be in the range from 0 to 1.
        """
        self.pixel_mask = pixel_mask
        self.mask_pyramid = get_gaussian_pyramid(self.pixel_mask, self.num_levels)

    def update_alphas(
        self, alpha_motion=None, alpha_residue=None, alpha_amplitude=None
    ):
        self.alpha = alpha_motion if alpha_motion else self.alpha
        self.alpha_residue = alpha_residue if alpha_residue else self.alpha_residue
        self.alpha_color = alpha_amplitude if alpha_amplitude else self.alpha_color

    def set_video(self, path_src: Union[str, Path]):
        """Read video to be magnified.

        Parameters
        ----------
        path_src : Union[str, Path]
            Path to the video to be magnified.
        """
        self._reset_frames()
        self._append_frames_in_path(path_src)

    def magnify_video(self, progress: bool = False) -> np.ndarray:
        """Run magnification algorithm on input video.

        Parameters
        ----------
        progress : bool, optional
            Show progress, by default False

        Returns
        -------
        np.ndarray
            Magnified video.
        """
        # Compute differences between adjacent monogenic phases.
        self._compute_differences(progress)

        # Temporally filter the differences and amplitudes
        self.apply_temporal_filter()

        # Spatially filter
        self._apply_amplitude_weighted_blur()

        # Use these differences to magnify frames
        frames_magnified = self._magnify_frames(
            self.frames_yiq, self.unwrapped_phase_i, self.unrapped_phase_j
        )
        return frames_magnified

    def _compute_differences(self, progress):
        n = len(self.frames_yiq)

        # Compute differences
        for i in range(n - 1):
            if progress:
                print("Computing difference {} of {}".format(i, n - 1))

            self._compute_difference_for_frame_i(i)

    def _compute_difference_for_frame_i(self, i):
        frame1, frame2 = self._get_frame_and_next_frame(i)
        self._compute_difference(frame1, frame2, idx=i)

    def _compute_difference(self, frame1: np.ndarray, frame2: np.ndarray, idx: int):
        """
        Compute differences in the logarithm of the monogenic representations
        between frame1 and frame2.

        Parameters
        ----------
        frame1 : np.ndarray
            Frame at position idx+1
        frame2 : np.ndarray
            Frame at position idx
        idx : int
            Index of frame 2
        """
        pass
        n = len(self.frames_yiq)

        pyramid_1 = self._compute_riesz_pyramid(frame1)
        lap_pyr_1, riesz_x_1, riesz_y_1 = pyramid_1.as_list_of_numpy()

        pyramid_2 = self._compute_riesz_pyramid(frame2)
        lap_pyr_2, riesz_x_2, riesz_y_2 = pyramid_2.as_list_of_numpy()

        # if None, then it is the first frames. So initialize memory.
        if self.count == 0:
            self.count += 1
            self._initialize_difference_computation_buffers(n, lap_pyr_1, lap_pyr_2)

        if self.temporal_filter_residue is not None:
            np.copyto(self.residues_original[idx], lap_pyr_2[-1])

        for k in range(0, self.num_levels):
            self._compute_difference_at_idx_and_level(
                idx, pyramid_1, pyramid_2, k
            )

    def _compute_difference_at_idx_and_level(
        self, idx, pyramid_1, pyramid_2, k
    ):
        self._compute_amplitude_at_idx_and_level(
            idx, pyramid_2, k
        )

        if self.temporal_filter_motion is not None:
            self._compute_complex_difference_at_idx_and_level(
                idx, pyramid_1, pyramid_2, k
            )

    def _compute_complex_difference_at_idx_and_level(
        self,
        idx: int,
        pyramid_1: RieszPyramid,
        pyramid_2: RieszPyramid,
        k: int,
    ):
        q_conj_prod_real, q_conj_prod_x, q_conj_prod_y = self._compute_q_conj_prod(
            pyramid_1, pyramid_2, k
        )
        q_conj_prod_amplitude = np.sqrt(
            q_conj_prod_real**2 + q_conj_prod_x**2 + q_conj_prod_y**2
        )
        ang = safe_divide(
            q_conj_prod_real, q_conj_prod_amplitude, self.min_val, self.rtol
        )
        phase_difference = np.arccos(ang)
        den = np.sqrt(q_conj_prod_x**2 + q_conj_prod_y**2)

        phase_diff_cos = phase_difference * q_conj_prod_x
        phase_diff_cos = safe_divide(phase_diff_cos, den, self.min_val, self.rtol)

        phase_diff_sin = phase_difference * q_conj_prod_y
        phase_diff_sin = safe_divide(phase_diff_sin, den, self.min_val, self.rtol)

        self.diffs_amplitude[k][idx, :, :] = np.sqrt(q_conj_prod_amplitude)
        self._set_phase_diffs_original_at_idx_and_level(
            idx, k, phase_diff_cos, phase_diff_sin
        )

    def _set_phase_diffs_original_at_idx_and_level(
        self, idx, k, phase_diff_cos, phase_diff_sin
    ):
        self.unwrapped_phase_i_original[k][idx, :, :] = phase_diff_cos
        self.unwrapped_phase_j_original[k][idx, :, :] = phase_diff_sin

        if len(self.unwrapped_phase_i_original[k]) > 0:
            self.unwrapped_phase_i_original[k][
                idx, :, :
            ] += self.unwrapped_phase_i_original[k][idx - 1, :, :]
            self.unwrapped_phase_j_original[k][
                idx, :, :
            ] += self.unwrapped_phase_j_original[k][idx - 1, :, :]

    def _compute_q_conj_prod(
        self, pyramid_1, pyramid_2, k
    ):
        lap_pyr_1, riesz_x_1, riesz_y_1 = pyramid_1.as_list_of_numpy()
        lap_pyr_2, riesz_x_2, riesz_y_2 = pyramid_2.as_list_of_numpy()

        if self.chr_att_motion == 0.0:
            q_conj_prod_real, q_conj_prod_x, q_conj_prod_y = compute_q_conj_prod(
                lap_pyr_1[k][:, :, 0],
                riesz_x_1[k][:, :, 0],
                riesz_y_1[k][:, :, 0],
                lap_pyr_2[k][:, :, 0],
                riesz_x_2[k][:, :, 0],
                riesz_y_2[k][:, :, 0],
            )
        else:
            q_conj_prod_real, q_conj_prod_x, q_conj_prod_y = compute_q_conj_prod(
                lap_pyr_1[k],
                riesz_x_1[k],
                riesz_y_1[k],
                lap_pyr_2[k],
                riesz_x_2[k],
                riesz_y_2[k],
            )

        return q_conj_prod_real, q_conj_prod_x, q_conj_prod_y

    def _compute_amplitude_at_idx_and_level(
        self, idx, pyramid, k
    ):
        lap_pyr_2, riesz_x_2, riesz_y_2 = pyramid.as_list_of_numpy()

        if self.temporal_filter_amplitude is not None:
            if self.chr_att_amplitude == 0.0:
                aux_lap_2 = lap_pyr_2[k][:, :, 0]
                aux_riesz_x_2 = riesz_x_2[k][:, :, 0]
                aux_riesz_y_2 = riesz_y_2[k][:, :, 0]
            else:
                aux_lap_2 = lap_pyr_2[k]
                aux_riesz_x_2 = riesz_x_2[k]
                aux_riesz_y_2 = riesz_y_2[k]

            self.amplitudes[k][idx] = np.sqrt(
                aux_lap_2[k] ** 2 + aux_riesz_x_2[k] ** 2 + aux_riesz_y_2[k] ** 2
            )

    def _initialize_difference_computation_buffers(
        self, n_frames, lap_pyr_1, lap_pyr_2
    ):
        if self.temporal_filter_motion is not None:
            self._init_frame_difference_buffers()

        if self.temporal_filter_amplitude is not None:
            self._init_amplitude_buffers()

        if self.temporal_filter_residue is not None:
            self._init_residue_buffers(n_frames, lap_pyr_2)

        for k in range(0, self.num_levels):
            self._set_initial_buffer_values_at_level(n_frames, lap_pyr_1, k)

    def _set_initial_buffer_values_at_level(self, n, lap_pyr_1, k):
        init_shapes_phases, init_shapes_amplitude = self._calculate_shapes_at_level(
            n, lap_pyr_1, k
        )

        if self.temporal_filter_motion is not None:
            self._append_frame_diff_initial_values(init_shapes_phases)

        if self.temporal_filter_amplitude is not None:
            self._append_amplitude_buffer_initial_values(init_shapes_amplitude)

    def _calculate_shapes_at_level(self, n, lap_pyr_1, k):
        if len(lap_pyr_1[k].shape) == 2 or self.chr_att_motion == 0.0:
            h, w = lap_pyr_1[k].shape[:2]
            init_shapes_phases = (n, h, w)
            init_shapes_amplitude = (n - 1, h, w)
        else:
            h, w, c = lap_pyr_1[k].shape
            init_shapes_phases = (n, h, w, c)
            init_shapes_amplitude = (n - 1, h, w, c)
        return init_shapes_phases, init_shapes_amplitude

    def _append_amplitude_buffer_initial_values(self, init_shapes_amplitude):
        self.amplitudes.append(np.zeros(init_shapes_amplitude, dtype=np.float32))

    def _append_frame_diff_initial_values(self, init_shapes_phases):
        self.unwrapped_phase_i.append(np.zeros(init_shapes_phases, dtype=np.float32))
        self.unrapped_phase_j.append(np.zeros(init_shapes_phases, dtype=np.float32))
        self.unwrapped_phase_i_original.append(
            np.zeros(init_shapes_phases, dtype=np.float32)
        )
        self.unwrapped_phase_j_original.append(
            np.zeros(init_shapes_phases, dtype=np.float32)
        )
        self.diffs_amplitude.append(np.zeros(init_shapes_phases, dtype=np.float32))

    def _init_residue_buffers(self, n, lap_pyr_2):
        if len(lap_pyr_2[-1].shape) == 2 or self.chr_att_residue == 0.0:
            h_res, w_res = lap_pyr_2[-1].shape
            self.residues = np.zeros((n - 1, h_res, w_res), dtype=np.float32)
        else:
            h_res, w_res, c_res = lap_pyr_2[-1].shape
            self.residues = np.zeros((n - 1, h_res, w_res, c_res), dtype=np.float32)
            self.residues_original = np.zeros(
                (n - 1, h_res, w_res, c_res), dtype=np.float32
            )

    def _init_amplitude_buffers(self):
        self.amplitudes = []

    def _init_frame_difference_buffers(self):
        self.unwrapped_phase_i = []
        self.unrapped_phase_j = []
        self.unwrapped_phase_i_original = []
        self.unwrapped_phase_j_original = []
        self.diffs_amplitude = []

    def _compute_riesz_pyramid(self, frame_y: np.ndarray) -> RieszPyramid:
        if self.ideal_riesz_transform:
            pyramid = RieszPyramid(
                frame=frame_y, n_levels=self.num_levels, algorithm="ideal"
            )
            # lap_pyr, riesz_x_pyr, riesz_y_pyr = pyramid.as_list_of_numpy()
        else:
            pyramid = RieszPyramid(
                frame=frame_y, n_levels=self.num_levels, algorithm="approximate"
            )
            # lap_pyr, riesz_x_pyr, riesz_y_pyr = pyramid.as_list_of_numpy()

        return pyramid

    def update_temporal_filters(
        self, 
        filter_motion: FullVideoTimeFilter = None, 
        filter_residue: FullVideoTimeFilter = None, 
        filter_amplitudes: FullVideoTimeFilter = None
    ):
        """Update the temporal filters used for the magnification algorithm.

        It is necessary to call `apply_temporal_filter` after updating the filters
        for them to have any effect.

        Parameters
        ----------
        filter_motion : FullVideoTimeFilter, optional
            Temporal filter applied over the phases to select motions, by default None
        filter_residue : FullVideoTimeFilter, optional
            Temporal filter applied over the residue to select color variations, by default None
        filter_amplitudes : FullVideoTimeFilter, optional
            Temporal filter applied over the amplitudes to select color variations, by default None
        """
        self.temporal_filter_motion = (
            filter_motion if filter_motion is not None else self.temporal_filter_motion
        )
        self.temporal_filter_residue = (
            filter_residue
            if filter_residue is not None
            else self.temporal_filter_residue
        )
        self.temporal_filter_amplitude = (
            filter_amplitudes
            if filter_amplitudes is not None
            else self.temporal_filter_amplitude
        )

    def apply_temporal_filter(self, batch_cols=10, progress=True):
        """Apply temporal filters over the decomposed video

        Parameters
        ----------
        batch_cols : int, optional
            Number of columns processed in a batch, by default 10
        progress : bool, optional
            Whether to show the progress, by default True
        """
        if self.temporal_filter_residue is not None:
            if progress:
                print("Applying temporal filter to the residue")
            self._filter_residue(batch_cols)

        if self.temporal_filter_motion is not None:
            if progress:
                print("Applying temporal filter to the phase")
            self._filter_differences(batch_cols)

        if self.temporal_filter_amplitude is not None:
            if progress:
                print("Applying temporal filter to the amplitudes")
            self._filter_amplitudes(batch_cols)

    def _filter_amplitudes(self, batch_cols):
        for k in range(0, self.num_levels):
            h, w = self.amplitudes[k].shape[1:3]
            self._filter_amplitudes_at_level(batch_cols, k, w)

    def _filter_amplitudes_at_level(self, batch_cols, k, w):
        for b in range(0, w, batch_cols):
            id0 = b
            idf = min(b + batch_cols, w)
            self._filter_amplitudes_at_columns_and_level(k, id0, idf)

    def _filter_differences(self, batch_cols):
        for k in range(0, self.num_levels):
            h, w = self.unwrapped_phase_i_original[k].shape[1:3]
            self._filter_differences_at_level(batch_cols, k, w)

    def _filter_differences_at_level(self, batch_cols, k, w):
        for b in range(0, w, batch_cols):
            id0 = b
            idf = min(b + batch_cols, w)
            self._filter_differences_at_columns_and_level(k, id0, idf)

    def _filter_amplitudes_at_columns_and_level(self, k, id0, idf):
        amplitudes_filtered = self.temporal_filter_amplitude.filter_array(
            self.amplitudes[k][:, :, id0:idf]
        )
        self.amplitudes[k][:, :, id0:idf] = safe_divide(
            amplitudes_filtered,
            self.amplitudes[k][:, :, id0:idf],
        )

    def _filter_differences_at_columns_and_level(self, k, id0, idf):
        self.unwrapped_phase_i[k][:, :, id0:idf] = (
            self.temporal_filter_motion.filter_array(
                self.unwrapped_phase_i_original[k][:, :, id0:idf]
            )
        )
        self.unrapped_phase_j[k][:, :, id0:idf] = (
            self.temporal_filter_motion.filter_array(
                self.unwrapped_phase_j_original[k][:, :, id0:idf]
            )
        )

    def _filter_residue(self, batch_cols):
        w = self.residues.shape[2]

        for b in range(0, w, batch_cols):
            id0 = b
            idf = min(b + batch_cols, w)
            self.residues[:, :, id0:idf, :] = self.temporal_filter_residue.filter_array(
                self.residues_original[:, :, id0:idf, :]
            )

    def _apply_amplitude_weighted_blur(self, progress=True):
        if self.temporal_filter_motion is not None:
            n = len(self.unwrapped_phase_i[0])
        else:
            return

        for i in range(n):
            if progress:
                print("Applying amplitude weighted blur to frame {} of {}".format(i, n))
                self._apply_amplitude_weighted_blur_to_idx(i)

    def _apply_amplitude_weighted_blur_to_idx(self, i):
        """
        Apply amplitude weighted blur to phases of frame i.

        Parameters
        ----------
        i : int
            Index of frame to apply blurring
        """
        for k in range(self.num_levels):
            self.unwrapped_phase_i[k][i, :, :] = amplitude_weighted_blur(
                self.unwrapped_phase_i[k][i, :, :],
                self.diffs_amplitude[k][i, :, :],
                self.ker_size,
                self.ker_sigma,
            )
            self.unrapped_phase_j[k][i, :, :] = amplitude_weighted_blur(
                self.unrapped_phase_j[k][i, :, :],
                self.diffs_amplitude[k][i, :, :],
                self.ker_size,
                self.ker_sigma,
            )

    def _magnify_frames(
        self, 
        frames_yiq: Sequence[np.ndarray], 
        filtered_phase_cos: Sequence[np.ndarray], 
        filtered_phase_sin: Sequence[np.ndarray]
    ) -> Sequence[np.ndarray]:
        """Shift phases of the channels of given frames according to the phase*cos 
        and phase*sin differences given.

        Parameters
        ----------
        frames_yiq : Sequence[np.ndarray]
            Frames in YIQ format to be magnified
        diffs_phase_cos : Sequence[np.ndarray]
            The i-th component of the monogenic phase filtered to contain only the 
            motion to be magnified
        diffs_phase_sin : Sequence[np.ndarray]
            The j-th component of the monogenic phase filtered to contain only the 
            motion to be magnified

        Returns
        -------
        Sequence[np.ndarray]
            Magnified frames
        """
        n = len(frames_yiq)
        frames_magnified = []
        for i in range(n):
            frame_mag = self._magnify_frame(
                frames_yiq, filtered_phase_cos, filtered_phase_sin, i, progress=True
            )
            frames_magnified.append(frame_mag)
        return frames_magnified

    def _magnify_frame(
        self, 
        frames_yiq, 
        filtered_phase_cos, 
        filtered_phase_sin, 
        i, 
        progress=True
    ):
        if progress:
            print("Shifting phase of frame {} of {}".format(i, len(frames_yiq)))
        
        frame_y = frames_yiq[i]
        pyramid = self._compute_riesz_pyramid(frame_y)
        lap, riesz_x, riesz_y = pyramid.as_list_of_numpy()

        lap_magnified = []  # build magnified real part of riesz pyramid
        for k in range(self.num_levels):
            self._magnify_frame_level(
                filtered_phase_cos, filtered_phase_sin, i, pyramid, lap_magnified, k
            )

        lap_magnified.append(lap[-1])  # append the unmodified residue
        if self.temporal_filter_residue is not None and i < self.residues.shape[0]:
            lap_magnified[-1] = (
                lap_magnified[-1]
                + self.alpha_residue * self.residues[i] * self.mask_pyramid[-1]
            )
        
        # collapse magnified
        frame_mag_y = self._collapse_laplacian_pyramid(lap_magnified)

        frame_mag = np.zeros(frames_yiq[i].shape, np.float32)
        if len(frame_mag_y.shape) == 2:  # only magnified luminance
            np.copyto(frame_mag, frames_yiq[i])
            np.copyto(frame_mag[:, :, 0], frame_mag_y)
        else:
            np.copyto(frame_mag, frame_mag_y)

        frame_mag = cv2.cvtColor(frame_mag, cv2.COLOR_YUV2RGB)
        frame_mag = np.clip(frame_mag, 0.0, 1.0)
        frame_mag *= 255.0
        return frame_mag.astype(np.uint8)

    def _magnify_frame_level(self, diffs_phase_cos, diffs_phase_sin, i, pyramid: RieszPyramid, lap_magnified, k):
        lap, _, _ = pyramid.as_list_of_numpy()
        if self.temporal_filter_motion is not None:
            magnified_band = self._compute_magnified_phases_band(
                    diffs_phase_cos, diffs_phase_sin, i, pyramid, k
                )
            lap_magnified.append(magnified_band)
        else:
            lap_magnified.append(lap[k])

            # magnify amplitude
        if self.temporal_filter_amplitude is not None:
            if i < self.amplitudes[k].shape[0]:                    
                    # amplitudes in this case is already normalized and temporally filtered
                magnified_amplitude_band = self._compute_magnified_amplitude_band(
                        i, lap_magnified, k
                    )

                lap_magnified[k] = magnified_amplitude_band

    def _compute_magnified_amplitude_band(self, i, lap_magnified, k):
        alpha_value = self.alpha_color if self.func_alpha is None else self.func_alpha(k, self.alpha_color)

        if alpha_value == 0:
            return lap_magnified[k]

        if len(lap_magnified[k].shape) == 3:
            magnified_luminance_amplitude_band = (
                1 + alpha_value * self.amplitudes[k][i, :, :] * self.mask_pyramid[k][:, :, 0]
            ) * lap_magnified[k][:, :, 0]

            magnified_amplitude_band = np.copy(lap_magnified[k])
            np.copyto(
                magnified_amplitude_band[:, :, 0], magnified_luminance_amplitude_band
            )
        else:
            magnified_amplitude_band = (
                1 + alpha_value * self.amplitudes[k][i, :, :] * self.mask_pyramid[k]
            ) * lap_magnified[k]
        
        return magnified_amplitude_band

    def _compute_magnified_phases_band(self, diffs_phase_cos, diffs_phase_sin, i, pyramid, k):
        lap, riesz_x, riesz_y = pyramid.as_list_of_numpy()
        if len(diffs_phase_cos[k].shape) == 3:  # no chrominance channels
            phase_filtered_cos = (
                diffs_phase_cos[k][i, :, :]
                * self.alpha
                * self.mask_pyramid[k][:, :, 0]
            )
            phase_filtered_sin = (
                diffs_phase_sin[k][i, :, :]
                * self.alpha
                * self.mask_pyramid[k][:, :, 0]
            )
        else:
            phase_filtered_cos = (
                diffs_phase_cos[k][i, :, :] * self.alpha * self.mask_pyramid[k]
            )
            phase_filtered_sin = (
                diffs_phase_sin[k][i, :, :] * self.alpha * self.mask_pyramid[k]
            )

        phase_mag = np.sqrt(phase_filtered_cos**2 + phase_filtered_sin**2)
        exp_phase_real = np.cos(phase_mag)

        num = phase_filtered_cos * np.sin(phase_mag)
        exp_phase_x = safe_divide(num, phase_mag)

        num = phase_filtered_sin * np.sin(phase_mag)
        exp_phase_y = safe_divide(num, phase_mag)

        if len(exp_phase_real.shape) == 3:
            magnified_band = exp_phase_real * lap[k] - exp_phase_x * riesz_x[k] - exp_phase_y * riesz_y[k]
        else:
            # Modify only luminance channel and copy chrominance channels from original pyramid
            magnified_band_luminance = (
                exp_phase_real * lap[k][:, :, 0]
                - exp_phase_x * riesz_x[k][:, :, 0]
                - exp_phase_y * riesz_y[k][:, :, 0]
            )
            magnified_band = np.zeros_like(lap[k])
            np.copyto(magnified_band[:, :, 0], magnified_band_luminance)
            np.copyto(magnified_band[:, :, 1:], lap[k][:, :, 1:])

        return magnified_band

    def _collapse_laplacian_pyramid(
        self, lap: Sequence[np.ndarray]
    ):
        """Collapse given pyramids

        Parameters
        ----------
        lap : Sequence[np.ndarray]
            Pyramid where lap[i] contains a numpy array with the frames from the i-th
            level of the pyramid. Last element is the top of the pyramid (residue).

        Returns
        -------
            Frames resulting from the collpsation
        """
        size_pyr = len(lap) - 1

        lap_up = cv2.pyrUp(lap[size_pyr], dstsize=lap[size_pyr - 1].shape[1::-1])
        lap_up += lap[size_pyr - 1]
        for i in range(size_pyr - 1, 1, -1):
            lap_up = cv2.pyrUp(lap_up, dstsize=lap[i - 1].shape[1::-1])
            lap_up += lap[i - 1]
        lap_up = cv2.pyrUp(lap_up, dstsize=lap[0].shape[1::-1])
        res = lap_up + lap[0]
        return res

    def _is_first_frame(self):
        return len(self.frames_yiq) == 0

    def _set_num_levels_in_pyramid_if_not_set(self, frame):
        if self._is_first_frame():
            h, w, c = frame.shape
            if self.num_levels is None:
                self.num_levels = max_pyramid_size(h, w) - 1

    def _rgb_to_32float_yuv(self, frame: np.ndarray):
        return cv2.cvtColor(frame.astype(np.float32) * 1.0 / 255, cv2.COLOR_RGB2YUV)

    def _set_masks_if_not_set(self):
        frame_example = self.frames_yiq[0]
        if self.mask_pyramid is None:
            if self.pixel_mask is None:
                self.pixel_mask = np.ones_like(frame_example)
            self.mask_pyramid = get_gaussian_pyramid(self.pixel_mask, self.num_levels)

    def _reset_frames(self):
        self.frames_yiq = []

    def _append_frames_in_path(self, path_src):
        path_src = str(Path(path_src).resolve())
        vc = cv2.VideoCapture(path_src)
        ret, frame = vc.read()
        while ret:
            self.append_frame(frame)
            ret, frame = vc.read()

    def _get_frame_and_next_frame(self, i):
        if self._no_chrominance_magnification():
            # 0 is the luminance channel: get only the luminance channel
            frame2 = self.frames_yiq[i][:, :, 0]
            frame1 = self.frames_yiq[i + 1][:, :, 0]
        else:
            frame2 = self.frames_yiq[i][:, :, :]
            frame1 = self.frames_yiq[i + 1][:, :, :]

        return frame1, frame2

    def _no_chrominance_magnification(self):
        return (
            self.chr_att_motion == 0.0
            and self.chr_att_amplitude == 0.0
            and self.chr_att_residue == 0.0
        )
    
    def _chorminance_magnification_is_set(self):
        return not self._no_chrominance_magnification()


class RieszAlphaFactor:
    """Class to create functions that return the alpha value given the level of the
    Riesz pyramid.
    """
    SELECTIVE = 0

    def __init__(
        self, 
        factor_type: Literal[0,], 
        num_levels: int, 
        which_levels=None
    ):
        """Initialize function creator object.

        Parameters
        ----------
        factor_type : Literal[0,]
            0 for the SELECTIVE function, which selects the levels to apply alpha to.
            The same value is applied to all of them.
        num_levels : int
            number of levels in the riesz pyramid without including the residue.
        which_levels : _type_, optional
            Mgnification factors specific to each level. Level 0 is the base of the 
            pyramid. If None, magnification is applied only to the last level with
            a value of `alpha`. By default None.
            The factors are multiplied by the given alpha to the function.
        """
        self.num_levels = num_levels
        self.factor_type = factor_type

        if factor_type == self.SELECTIVE:
            if which_levels is None:
                self.which_levels = [0 for _ in range(self.num_levels)]
                self.which_levels[self.num_levels - 1] = 1
            else:
                self.which_levels = which_levels

    def get_function(self):
        if self.factor_type == self.SELECTIVE:
            return self.select_levels_factor

    def select_levels_factor(self, level, alpha):
        return self.which_levels[level] * alpha