from pathlib import Path
from typing import Union
import numpy as np
import cv2

from rieszsimultaneous.utils.processing import (
    amplitude_weighted_blur,
    get_gaussian_pyramid,
    max_pyramid_size,
)


class RieszMagnificationFullVideo:
    def __init__(
        self,
        num_levels=None,
        alpha=50,
        temporal_filter_motion=None,
        temporal_filter_amplitude=None,
        temporal_filter_residue=None,
        alpha_amplitude=None,
        alpha_residue=None,
        func_alpha=None,
        kernel_sigma=2,
        ideal_riesz_transform=False,
        chr_attenuation_motion=0.0,
        chr_attenuation_amplitude=0.0,
        chr_attenuation_residue=1.0,
        pixel_mask=None,
        single_step_filter=False,
    ):
        """
        Apply Riesz motion and color-changes magnification.

        :param num_levels: number of levels in the Riesz pyramid, including the residue. If None, uses max size.
        :param alpha: magnification factor
        :param temporal_filter_motion: temporal filter for the phases of the monogenic signal
        :param temporal_filter_amplitude: temporal filter for the amplitudes of the monogenic signal
        :param temporal_filter_residue: temporal filter for the (low-pass) residue of the pyramid
        :param ideal_riesz_transform: use ideal riesz transform in frequency instead of approximation
        :param chr_attenuation_motion: chrominance attenuation factor applied to motion magnification
        :param chr_attenuation_residue: chrominance attenuation factor applied to color-changes magnification
        :param pixel_mask: ndarray with same dimensions as the input frame. The mask is decomposed in a gaussian pyramid
        and it is used at each level in order to select the region. Values should be in range from 0 to 1.
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
        self.diffs_phase_cos = None
        self.diffs_phase_sin = None
        self.diffs_phase_cos_original = None
        self.diffs_phase_sin_original = None
        self.diffs_amplitude = None
        self.amplitudes = None
        self.residues = None
        self.residues_original = None
        self.kernel_x = None
        self.kernel_y = None
        self.ker_size = 0
        self.ker_sigma = kernel_sigma
        self.min_val = 1e-9
        self.rtol = 1e-5
        self.init_kernels()

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

    def update_pixel_mask(self, pixel_mask):
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
        self.compute_differences(progress)

        # Temporally filter the differences and amplitudes
        self.apply_temporal_filter()

        # Spatially filter
        self.apply_amplitude_weighted_blur()

        # Use these differences to magnify frames
        frames_magnified = self.magnify_frames(
            self.frames_yiq, self.diffs_phase_cos, self.diffs_phase_sin
        )
        return frames_magnified

    def compute_differences(self, progress):
        n = len(self.frames_yiq)

        # Compute differences
        for i in range(n - 1):
            if progress:
                print("Computing difference {} of {}".format(i, n - 1))

            self.compute_difference_for_frame_i(i)

    def compute_difference_for_frame_i(self, i):
        frame1, frame2 = self._get_frame_and_next_frame(i)
        self.compute_difference(frame1, frame2, idx=i)

    def init_kernels(self):
        self.kernel_x = np.zeros((3, 3), dtype=np.float32)
        self.kernel_x[1, 0] = 0.5
        self.kernel_x[1, 2] = -0.5

        self.kernel_y = np.zeros((3, 3), dtype=np.float32)
        self.kernel_y[0, 1] = 0.5
        self.kernel_y[2, 1] = -0.5

    def compute_difference(self, frame1: np.ndarray, frame2: np.ndarray, idx: int):
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
        lap_pyr_1, riesz_x_1, riesz_y_1 = self.compute_riesz_pyramid(frame1)
        lap_pyr_2, riesz_x_2, riesz_y_2 = self.compute_riesz_pyramid(frame2)

        # if None, then it is the first frames. So initialize memory.
        if self.count == 0:
            self.count += 1
            self._initialize_difference_computation_buffers(n, lap_pyr_1, lap_pyr_2)

        if self.temporal_filter_residue is not None:
            np.copyto(self.residues_original[idx], lap_pyr_2[-1])

        for k in range(0, self.num_levels):
            self._compute_difference_at_idx_and_level(
                idx, lap_pyr_1, riesz_x_1, riesz_y_1, lap_pyr_2, riesz_x_2, riesz_y_2, k
            )

    def _compute_difference_at_idx_and_level(
        self, idx, lap_pyr_1, riesz_x_1, riesz_y_1, lap_pyr_2, riesz_x_2, riesz_y_2, k
    ):
        self._compute_amplitude_at_idx_and_level(idx, lap_pyr_2, riesz_x_2, riesz_y_2, k)

        if self.temporal_filter_motion is not None:
            if self.chr_att_motion == 0.0:
                q_conj_prod_real, q_conj_prod_x, q_conj_prod_y = self._compute_q_conj_prod(
                    lap_pyr_1[k][:, :, 0], riesz_x_1[k][:, :, 0], riesz_y_1[k][:, :, 0], 
                    lap_pyr_2[k][:, :, 0], riesz_x_2[k][:, :, 0], riesz_y_2[k][:, :, 0],
                )
            else:
                q_conj_prod_real, q_conj_prod_x, q_conj_prod_y = self._compute_q_conj_prod(
                    lap_pyr_1[k], riesz_x_1[k], riesz_y_1[k], lap_pyr_2[k], riesz_x_2[k], riesz_y_2[k]
                )

            q_conj_prod_amplitude = np.sqrt(
                q_conj_prod_real**2 + q_conj_prod_x**2 + q_conj_prod_y**2
            )
            self.diffs_amplitude[k][idx, :, :] = np.sqrt(q_conj_prod_amplitude)

            ang = np.divide(
                q_conj_prod_real,
                q_conj_prod_amplitude,
                out=np.zeros_like(q_conj_prod_real),
                where=~np.isclose(
                    q_conj_prod_amplitude, 0, rtol=self.rtol, atol=self.min_val
                ),
            )
            phase_difference = np.arccos(ang)
            den = np.sqrt(q_conj_prod_x**2 + q_conj_prod_y**2)
            phase_diff_cos = phase_difference * q_conj_prod_x
            phase_diff_cos = np.divide(
                phase_diff_cos,
                den,
                out=np.zeros_like(phase_diff_cos),
                where=~np.isclose(den, 0, rtol=self.rtol, atol=self.min_val),
            )
            phase_diff_sin = phase_difference * q_conj_prod_y
            phase_diff_sin = np.divide(
                phase_diff_sin,
                den,
                out=np.zeros_like(phase_diff_sin),
                where=~np.isclose(den, 0, rtol=self.rtol, atol=self.min_val),
            )
            if len(self.diffs_phase_cos_original[k]) == 0:
                self.diffs_phase_cos_original[k][idx, :, :] = phase_diff_cos
                self.diffs_phase_sin_original[k][idx, :, :] = phase_diff_sin
            else:
                self.diffs_phase_cos_original[k][idx, :, :] = (
                    self.diffs_phase_cos_original[k][idx - 1, :, :] + phase_diff_cos
                )
                self.diffs_phase_sin_original[k][idx, :, :] = (
                    self.diffs_phase_sin_original[k][idx - 1, :, :] + phase_diff_sin
                )

    def _compute_q_conj_prod(self, lap_pyr_1, riesz_x_1, riesz_y_1, lap_pyr_2, riesz_x_2, riesz_y_2):
        q_conj_prod_real = (
            lap_pyr_1 * lap_pyr_2
            + riesz_x_1 * riesz_x_2
            + riesz_y_1 * riesz_y_2
        )
        q_conj_prod_x = (
            -lap_pyr_1 * riesz_x_2 + lap_pyr_2 * riesz_x_1
        )
        q_conj_prod_y = (
            -lap_pyr_1 * riesz_y_2 + lap_pyr_2 * riesz_y_1
        )
        
        return q_conj_prod_real, q_conj_prod_x, q_conj_prod_y

    def _compute_amplitude_at_idx_and_level(self, idx, lap_pyr_2, riesz_x_2, riesz_y_2, k):
        if self.temporal_filter_amplitude is not None:
            self.amplitudes[k][idx] = np.sqrt(
                lap_pyr_2[k] ** 2 + riesz_x_2[k] ** 2 + riesz_y_2[k] ** 2
            )

    def _initialize_difference_computation_buffers(self, n, lap_pyr_1, lap_pyr_2):
        if self.temporal_filter_motion is not None:
            self.diffs_phase_cos = []
            self.diffs_phase_sin = []
            self.diffs_phase_cos_original = []
            self.diffs_phase_sin_original = []
            self.diffs_amplitude = []
        if self.temporal_filter_amplitude is not None:
            self.amplitudes = []
        if self.temporal_filter_residue is not None:
            if len(lap_pyr_2[-1].shape) == 2 or self.chr_att_residue == 0.0:
                h_res, w_res = lap_pyr_2[-1].shape
                self.residues = np.zeros((n - 1, h_res, w_res), dtype=np.float32)
            else:
                h_res, w_res, c_res = lap_pyr_2[-1].shape
                self.residues = np.zeros((n - 1, h_res, w_res, c_res), dtype=np.float32)
                self.residues_original = np.zeros(
                    (n - 1, h_res, w_res, c_res), dtype=np.float32
                )

        for k in range(0, self.num_levels):
            if len(lap_pyr_1[k].shape) == 2 or self.chr_att_motion == 0.0:
                h, w = lap_pyr_1[k].shape[:2]
                init_shapes_phases = (n, h, w)
                init_shapes_amplitude = (n - 1, h, w)
            else:
                h, w, c = lap_pyr_1[k].shape
                init_shapes_phases = (n, h, w, c)
                init_shapes_amplitude = (n - 1, h, w, c)

            print("Shapes phases: {}".format(init_shapes_phases))
            if self.temporal_filter_motion is not None:
                self.diffs_phase_cos.append(
                    np.zeros(init_shapes_phases, dtype=np.float32)
                )
                self.diffs_phase_sin.append(
                    np.zeros(init_shapes_phases, dtype=np.float32)
                )
                self.diffs_phase_cos_original.append(
                    np.zeros(init_shapes_phases, dtype=np.float32)
                )
                self.diffs_phase_sin_original.append(
                    np.zeros(init_shapes_phases, dtype=np.float32)
                )

                self.diffs_amplitude.append(
                    np.zeros(init_shapes_phases, dtype=np.float32)
                )
            if self.temporal_filter_amplitude is not None:
                self.amplitudes.append(
                    np.zeros(init_shapes_amplitude, dtype=np.float32)
                )

    def compute_riesz_pyramid(self, frame_y):
        """
        Compute riesz pyramid of given frame. Index 0 is the base of the pyramid. Real part has one more element
        which corresponds to the residue.
        :param frame_y: frame to use as the base of the pyramid
        :return: real, riesz_x, riesz_y
        """
        lap_pyr = [np.zeros(frame_y.shape, dtype=np.float32)]
        np.copyto(lap_pyr[0], frame_y)
        riesz_x_pyr = []
        riesz_y_pyr = []
        for k in range(0, self.num_levels):
            # construct level k by computing its reduced version, expanding it and subtracting it
            lap_down = cv2.pyrDown(lap_pyr[k])
            lap_pyr.append(lap_down)
            lap_up = cv2.pyrUp(lap_down, dstsize=lap_pyr[k].shape[1::-1])
            lap_pyr[k] -= lap_up

            if not self.ideal_riesz_transform:
                riesz_x_pyr.append(cv2.filter2D(lap_pyr[k], -1, self.kernel_x))
                riesz_y_pyr.append(cv2.filter2D(lap_pyr[k], -1, self.kernel_y))
            else:
                im_riesz_x, im_riesz_y = compute_ideal_riesz_transform(lap_pyr[k])
                riesz_x_pyr.append(im_riesz_x)
                riesz_y_pyr.append(im_riesz_y)

        return lap_pyr, riesz_x_pyr, riesz_y_pyr

    def update_temporal_filters(
        self, filter_motion=None, filter_residue=None, filter_amplitudes=None
    ):
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
        if (
            self.temporal_filter_residue is not None
        ):  # apply temporal filter to residue of pyramids
            if len(self.residues[0]) == 2:
                h, w = self.residues[0].shape
            else:
                h, w, c = self.residues[0].shape

            num_batches = w // batch_cols
            last_batch_size = w % batch_cols
            for b in range(0, num_batches):
                id0 = batch_cols * b
                idf = batch_cols * (b + 1)
                self.residues[
                    :, :, id0:idf, :
                ] = self.temporal_filter_residue.filter_array(
                    self.residues_original[:, :, id0:idf, :]
                )

            if last_batch_size != 0:
                id0 = batch_cols * num_batches
                print(self.residues[:, :, id0].shape)
                self.residues[
                    :, :, id0:, :
                ] = self.temporal_filter_residue.filter_array(
                    self.residues_original[:, :, id0:, :]
                )

        if (
            self.temporal_filter_motion is None
            and self.temporal_filter_amplitude is None
        ):
            return

        for k in range(0, self.num_levels):
            if progress:
                print("Filtering level {} of {}".format(k, self.num_levels))

            if (
                self.temporal_filter_motion is not None
                or self.temporal_filter_amplitude is not None
            ):
                if self.temporal_filter_motion is not None:
                    array_template = self.diffs_phase_cos_original[k][0, :, :]
                    h, w = self.diffs_phase_cos_original[k].shape[1:3]
                else:
                    # array_template = self.amplitudes[k][0, :, :]
                    h, w = self.amplitudes[k].shape[1:3]

                # if len(array_template[0]) == 2:
                #     h, w = array_template.shape
                # else:
                #     h, w, c = array_template.shape

                num_batches = w // batch_cols
                last_batch_size = w % batch_cols
                for b in range(0, num_batches):
                    id0 = batch_cols * b
                    idf = batch_cols * (b + 1)
                    # filter log of differences of monogenic signals
                    if self.temporal_filter_motion is not None:
                        self.diffs_phase_cos[k][
                            :, :, id0:idf
                        ] = self.temporal_filter_motion.filter_array(
                            self.diffs_phase_cos_original[k][:, :, id0:idf]
                        )
                        self.diffs_phase_sin[k][
                            :, :, id0:idf
                        ] = self.temporal_filter_motion.filter_array(
                            self.diffs_phase_sin_original[k][:, :, id0:idf]
                        )
                    # filter amplitudes of monogenic signals
                    if self.temporal_filter_amplitude is not None:
                        amplitudes_filtered = (
                            self.temporal_filter_amplitude.filter_array(
                                self.amplitudes[k][:, :, id0:idf]
                            )
                        )
                        self.amplitudes[k][:, :, id0:idf] = np.divide(
                            amplitudes_filtered,
                            self.amplitudes[k][:, :, id0:idf],
                            out=np.zeros_like(amplitudes_filtered),
                            where=~np.isclose(
                                self.amplitudes[k][:, :, id0:idf],
                                0,
                                rtol=self.rtol,
                                atol=self.min_val,
                            ),
                        )

                if last_batch_size != 0:
                    id0 = batch_cols * num_batches
                    # filter log of differences of monogenic signals
                    if self.temporal_filter_motion is not None:
                        self.diffs_phase_cos[k][
                            :, :, id0:
                        ] = self.temporal_filter_motion.filter_array(
                            self.diffs_phase_cos_original[k][:, :, id0:]
                        )
                        self.diffs_phase_sin[k][
                            :, :, id0:
                        ] = self.temporal_filter_motion.filter_array(
                            self.diffs_phase_sin_original[k][:, :, id0:]
                        )
                    # filter amplitudes of monogenic signals
                    if self.temporal_filter_amplitude is not None:
                        amplitudes_filtered = (
                            self.temporal_filter_amplitude.filter_array(
                                self.amplitudes[k][:, :, id0:]
                            )
                        )
                        self.amplitudes[k][:, :, id0:] = np.divide(
                            amplitudes_filtered,
                            self.amplitudes[k][:, :, id0:],
                            out=np.zeros_like(amplitudes_filtered),
                            where=~np.isclose(
                                self.amplitudes[k][:, :, id0:],
                                0,
                                rtol=self.rtol,
                                atol=self.min_val,
                            ),
                        )

    def apply_amplitude_weighted_blur(self, progress=True):
        if self.temporal_filter_motion is not None:
            n = len(self.diffs_phase_cos[0])
        else:
            return

        for i in range(n):
            if progress:
                print("Applying amplitude weighted blur to frame {} of {}".format(i, n))
                self.apply_amplitude_weighted_blur_to_idx(i)

    def apply_amplitude_weighted_blur_to_idx(self, i):
        """
        Apply amplitude weighted blur to phases of frame i.

        Parameters
        ----------
        i : int
            Index of frame to apply blurring
        """
        for k in range(self.num_levels):
            self.diffs_phase_cos[k][i, :, :] = amplitude_weighted_blur(
                self.diffs_phase_cos[k][i, :, :],
                self.diffs_amplitude[k][i, :, :],
                self.ker_size,
                self.ker_sigma,
            )
            self.diffs_phase_sin[k][i, :, :] = amplitude_weighted_blur(
                self.diffs_phase_sin[k][i, :, :],
                self.diffs_amplitude[k][i, :, :],
                self.ker_size,
                self.ker_sigma,
            )

    def magnify_frames(
        self, frames_yiq, diffs_phase_cos, diffs_phase_sin, progress=True
    ):
        """
        Shift phases of luminance channel of given frames according to the phase*cos and phase*sin differences given.
        If amplitude and residue temporal filters were given, also magnify them.

        :param frames_yiq: list of yiq frames
        :param diffs_phase_cos: list in format [levels][time, height, col] used to magnify luminance channel
        :param diffs_phase_sin: list in format [levels][time, height, col] used to magnify luminance channel
        :param progress: print progress
        :return: list of yiq magnified frames
        """
        min_val = self.min_val
        n = len(frames_yiq)
        frames_magnified = []
        for i in range(n):
            frame_mag = self.magnify_frame(
                frames_yiq, diffs_phase_cos, diffs_phase_sin, i, progress=True
            )
            frames_magnified.append(frame_mag)
        return frames_magnified

    def magnify_frame(
        self, frames_yiq, diffs_phase_cos, diffs_phase_sin, i, progress=True
    ):
        if progress:
            print("Shifting phase of frame {} of {}".format(i, len(frames_yiq)))
        frame_y = frames_yiq[i]
        lap, riesz_x, riesz_y = self.compute_riesz_pyramid(frame_y)

        lap_magnified = []  # build magnified real part of riesz pyramid
        for k in range(self.num_levels):
            if self.temporal_filter_motion is not None:
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
                exp_phase_x = np.divide(
                    num,
                    phase_mag,
                    out=np.zeros_like(num),
                    where=~np.isclose(phase_mag, 0, rtol=self.rtol, atol=self.min_val),
                )
                num = phase_filtered_sin * np.sin(phase_mag)
                exp_phase_y = np.divide(
                    num,
                    phase_mag,
                    out=np.zeros_like(num),
                    where=~np.isclose(phase_mag, 0, rtol=self.rtol, atol=self.min_val),
                )

                if len(exp_phase_real.shape) == 3:
                    lap_magnified.append(
                        exp_phase_real * lap[k]
                        - exp_phase_x * riesz_x[k]
                        - exp_phase_y * riesz_y[k]
                    )
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

                    lap_magnified.append(magnified_band)
            else:
                lap_magnified.append(lap[k])

            # magnify amplitude
            if (
                self.temporal_filter_amplitude is not None
                and i < self.amplitudes[k].shape[0]
            ):
                # amplitudes in this case is already normalized and temporally filtered
                if self.func_alpha is None:
                    lap_magnified[k] = (
                        1 + self.alpha_color * self.amplitudes[k][i, :, :]
                    ) * lap_magnified[k]
                else:
                    # print("Amplitude mag: {}".format(self.func_alpha(k, self.alpha_color)))
                    lap_magnified[k] = (
                        1
                        + self.func_alpha(k, self.alpha_color)
                        * self.amplitudes[k][i, :, :]
                    ) * lap_magnified[k]

        lap_magnified.append(lap[-1])  # append the unmodified residue
        if self.temporal_filter_residue is not None and i < self.residues.shape[0]:
            lap_magnified[-1] = (
                lap_magnified[-1]
                + self.alpha_residue * self.residues[i] * self.mask_pyramid[-1]
            )

        # collapse magnified
        frame_mag_y = self.collapse_laplacian_pyramid(lap_magnified)

        # if self.temporal_filter_residue is not None and i < self.residues.shape[0]:
        #     _, pyr_sizes = sm.build_gaussian_pyramid(frames_yiq[0], self.num_levels, return_sizes=True)
        #     collapsed_residue = sm.collapse_gaussian_pyramid(self.alpha_residue * self.residues[i], pyr_sizes)
        #     # cv2.imshow("collapsed_residue", collapsed_residue)
        #     # cv2.waitKey(0)
        #     frame_mag_y += collapsed_residue

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

    def collapse_laplacian_pyramid(self, lap):
        """
        Collapse given pyramid and return resulting frame.

        :param lap: laplacian pyramid. Last element is the top of the pyramid (residue)
        :return: resulting frame
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


def compute_ideal_riesz_transform(im):
    """
    Compute Riesz Transform in frequency.

    :param im: numpy 2-D array of which to compute the riesz transform
    :return: pair of numpy arrays (riesz_x, riesz_y)
    """
    if len(im.shape) != 2:
        raise Exception("Im be a 2-d numpy array")

    # guarantee symmetric axes around zero
    h, w = im.shape
    h_padded = h if h % 2 != 0 else h + 1
    w_padded = w if w % 2 != 0 else w + 1

    # frequency space
    fx = np.arange(-(w_padded - 1) // 2, w_padded // 2 + 1) / w_padded
    fy = np.arange(-(h_padded - 1) // 2, h_padded // 2 + 1) / h_padded
    mfx, mfy = np.meshgrid(fx, fy)
    r = np.sqrt(mfx**2 + mfy**2)

    # frequency responses of riesz transform
    mask = ~np.isclose(r, 0, rtol=1e-7, atol=1e-11)
    rx = -1j * np.divide(mfx, r, out=np.ones_like(mfx), where=mask)
    ry = -1j * np.divide(mfy, r, out=np.ones_like(mfx), where=mask)
    # rx = -1j * mfx / r
    # ry = -1j * mfy / r

    # compute transform of im

    im_fft = np.fft.fftshift(np.fft.fft2(im, s=[h_padded, w_padded]))
    im_rx_fft = im_fft * rx
    im_ry_fft = im_fft * ry

    # return to spatial domain
    im_rx = np.fft.ifft2(np.fft.ifftshift(im_rx_fft))
    im_ry = np.fft.ifft2(np.fft.ifftshift(im_ry_fft))
    im_rx = np.real(im_rx[:h, :w])  # remove extra dimension used for symmetry
    im_ry = np.real(im_ry[:h, :w])  # remove extra dimension used for symmetry

    return im_rx, im_ry
