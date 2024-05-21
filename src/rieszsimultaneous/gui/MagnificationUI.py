from typing import Tuple
import cv2
import numpy as np
import logging

from rieszsimultaneous.algorithms import RieszMagnificationFullVideo
from rieszsimultaneous.algorithms.riesz import RieszAlphaFactor
from rieszsimultaneous.filters import ButterworthFullVideo
from rieszsimultaneous.masks import chrominance_similarity_mask


logger = logging.getLogger("my_logger")
logger.setLevel(
    logging.DEBUG
)


class MagnificationManager:
    VIDEO_WINDOW_NAME = "Video"
    PARAMETERS_WINDOW_NAME = "Parameters"
    MASK_WINDOW_NAME = "Mask"
    ALPHA_MOTION_TRACKBAR_NAME = "Motion Magnification"
    ALPHA_COLOR_TRACKBAR_NAME = "Color-change Magnification"

    FILTER_MOTION_LOW_TRACKBAR_NAME = "Low cut-off Frequency Motion"
    FILTER_MOTION_HIGH_TRACKBAR_NAME = "High cut-off Frequency Motion"
    FILTER_COLOR_LOW_TRACKBAR_NAME = "Low cut-off Frequency Color Change"
    FILTER_COLOR_HIGH_TRACKBAR_NAME = "High cut-off Frequency Color Change"
    UPDATE_FILTERS_BUTTON_NAME = "Update Temporal Filter"

    ONLY_MOTION_RADIO_BUTTON_NAME = "Motion"
    ONLY_COLOR_RADIO_BUTTON_NAME = "Color changes"
    BOTH_RADIO_BUTTON_NAME = "Both"

    def __init__(
        self,
        alpha_motion: float,
        alpha_color: float,
        band_motion: Tuple[float, float],
        band_color: Tuple[float, float],
        mask: np.ndarray = None,
        alpha_motion_max=100,
        alpha_color_max=300,
        freq_max=15.0,
    ):
        """Initialize magnification algorithm with initial values

        Parameters
        ----------
        alpha_motion : float
            Magnification factor for the motion
        alpha_color : float
            Magnification factor for the color variations
        band_motion : Tuple[float, float]
            Low and high cut-off frequencies for the motion magnification
        band_color : Tuple[float, float]
            Low and high cut-off frequencies for the color magnification
        mask : np.ndarray, optional
            Pixel mask as defined in the magnification algorithm, by default None
        alpha_motion_max : int, optional
            Maximum alpha value for motion in the slider, by default 100
        alpha_color_max : int, optional
            Maximum alpha value for color in the slider, by default 300
        freq_max : float, optional
            Maximum frequency values in the sliders, by default 15.0
        """
        self.alpha_motion = alpha_motion
        self.alpha_color = alpha_color
        self.alpha_motion_max = alpha_motion_max
        self.alpha_color_max = alpha_color_max
        self.band_motion = band_motion
        self.band_color = band_color
        self.freq_max = freq_max
        self.mask = mask

        self.apply_blur = True
        self.path_src = None
        self.mag_algorithm = None
        self.filter_motion = None
        self.filter_color = None
        self.video_capture = None
        self.frame_rate = None
        self.playing = False
        self.frame0 = None
        self.masks = []

    def start_gui_video(self, path_src: str):
        """
        Start GUI for given video.
        Algorithms are applied with given initial parameters, but they can be updated using the update methods.

        Parameters
        ----------
        path_src : str
            Path to video to analyse.
        """
        self.path_src = path_src
        self.video_capture = cv2.VideoCapture(self.path_src)
        ret, self.frame0 = self.video_capture.read()
        self.frame_rate = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.video_capture.release()

        self.filter_motion = ButterworthFullVideo(
            self.frame_rate, self.band_motion[0], self.band_motion[1], order=2
        )
        self.filter_color = ButterworthFullVideo(
            self.frame_rate, self.band_color[0], self.band_color[1], order=2
        )
        self.filter_color_amplitudes = ButterworthFullVideo(
            self.frame_rate, self.band_color[0], self.band_color[1], order=2
        )
        pixel_mask = np.zeros_like(self.frame0)

        self.initialize_magnification_algorithm(pixel_mask)
        self.preprocess_video()

        # create windows
        self.create_gui_windows()

        while True:
            self.apply_blur = True

            for i in range(0, len(self.mag_algorithm.frames_yiq)):
                logger.info("Frame {}".format(i))

                if self.apply_blur:
                    self.mag_algorithm._apply_amplitude_weighted_blur_to_idx(i)

                frame_mag = self.mag_algorithm._magnify_frame(
                    self.mag_algorithm.frames_yiq,
                    self.mag_algorithm.unwrapped_phase_i,
                    self.mag_algorithm.unrapped_phase_j,
                    i,
                )

                cv2.imshow(self.VIDEO_WINDOW_NAME, frame_mag)
                if self.playing:
                    key = cv2.waitKey(int(1000 / self.frame_rate))
                else:
                    key = cv2.waitKey(0)

                self.on_key_pressed(key)

            self.apply_blur = False

    def create_gui_windows(self):
        cv2.namedWindow(self.VIDEO_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(self.VIDEO_WINDOW_NAME, self.on_mouse_click)

        cv2.namedWindow(self.PARAMETERS_WINDOW_NAME, cv2.WINDOW_GUI_NORMAL)
        cv2.createTrackbar(
            self.ALPHA_MOTION_TRACKBAR_NAME,
            self.PARAMETERS_WINDOW_NAME,
            self.alpha_motion,
            self.alpha_motion_max,
            self.on_trackbar_motion,
        )
        cv2.createTrackbar(
            self.ALPHA_COLOR_TRACKBAR_NAME,
            self.PARAMETERS_WINDOW_NAME,
            self.alpha_color,
            self.alpha_color_max,
            self.on_trackbar_color,
        )

        cv2.createTrackbar(
            self.FILTER_MOTION_LOW_TRACKBAR_NAME,
            self.PARAMETERS_WINDOW_NAME,
            int(self.band_motion[0] * 100),
            int(self.freq_max * 100),
            self.on_trackbar_low_freq_motion,
        )
        cv2.createTrackbar(
            self.FILTER_MOTION_HIGH_TRACKBAR_NAME,
            self.PARAMETERS_WINDOW_NAME,
            int(self.band_motion[1] * 100),
            int(self.freq_max * 100),
            self.on_trackbar_high_freq_motion,
        )
        cv2.createTrackbar(
            self.FILTER_COLOR_LOW_TRACKBAR_NAME,
            self.PARAMETERS_WINDOW_NAME,
            int(self.band_color[0] * 100),
            int(self.freq_max * 100),
            self.on_trackbar_low_freq_color,
        )
        cv2.createTrackbar(
            self.FILTER_COLOR_HIGH_TRACKBAR_NAME,
            self.PARAMETERS_WINDOW_NAME,
            int(self.band_color[1] * 100),
            int(self.freq_max * 100),
            self.on_trackbar_high_freq_color,
        )
        cv2.namedWindow(self.MASK_WINDOW_NAME)

    def initialize_magnification_algorithm(self, pixel_mask):
        num_levels = 7
        which_levels = [0 for _ in range(num_levels - 1)]
        which_levels[-1] = 1
        func_alpha = RieszAlphaFactor(
            RieszAlphaFactor.SELECTIVE, num_levels, which_levels=which_levels
        )
        self.mag_algorithm = RieszMagnificationFullVideo(
            num_levels=7,
            alpha=self.alpha_motion,
            temporal_filter_motion=self.filter_motion,
            alpha_residue=self.alpha_color,
            temporal_filter_residue=self.filter_color,
            alpha_amplitude=self.alpha_color/2,
            temporal_filter_amplitude=self.filter_color_amplitudes,
            pixel_mask=pixel_mask,
            func_alpha=func_alpha.get_function(),
        )

    def on_key_pressed(self, key):
        if key == ord("p"):
            self.playing = not self.playing
        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit(0)
        elif key == ord("t"):
            self.update_temporal_filter()

    def on_mouse_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.update_mask((x, y))

    def on_trackbar_motion(self, val):
        self.mag_algorithm.update_alphas(alpha_motion=float(val))

    def on_trackbar_color(self, val):
        self.mag_algorithm.update_alphas(alpha_residue=float(val))

    def update_mask(self, p0, threshold=0.08):
        logger.info("Updating masks")

        new_mask = True
        x, y = p0
        for mask_idx, current_mask in enumerate(self.masks, 0):
            if current_mask[y, x, 0] == 1:
                logger.info("Removing mask over pixels similar to {}".format(x, y))
                del self.masks[mask_idx]
                new_mask = False

        if new_mask:
            self.masks.append(
                chrominance_similarity_mask(self.frame0, p0, threshold=threshold)
            )

        full_mask = np.zeros_like(self.mag_algorithm.frames_yiq[0])
        for m in self.masks:
            full_mask = np.clip(((full_mask > 0.99) + (m > 0.99)), 0.0, 1.0)

        cv2.imshow(self.MASK_WINDOW_NAME, full_mask)

        self.mag_algorithm.update_pixel_mask(full_mask)

    def on_trackbar_low_freq_motion(self, val):
        self.band_motion = (val / 100.0, self.band_motion[1])

    def on_trackbar_high_freq_motion(self, val):
        self.band_motion = (self.band_motion[0], val / 100.0)

    def on_trackbar_low_freq_color(self, val):
        self.band_color = (val / 100.0, self.band_color[1])

    def on_trackbar_high_freq_color(self, val):
        self.band_color = (self.band_color[0], val / 100.0)

    def update_temporal_filter(self):
        """
        Update temporal filter with current band_motion and band_color attributes and 
        re-apply filters to video.
        """
        logger.info("Updating temporal filters...")
        self.filter_motion = ButterworthFullVideo(
            self.frame_rate, self.band_motion[0], self.band_motion[1], order=2
        )
        self.filter_color = ButterworthFullVideo(
            self.frame_rate, self.band_color[0], self.band_color[1], order=2
        )

        self.mag_algorithm.update_temporal_filters(
            filter_motion=self.filter_motion, filter_residue=self.filter_color
        )
        self.mag_algorithm.apply_temporal_filter()
        self.apply_blur = True

        logger.info("Done.")

    def preprocess_video(self):
        logger.info("Preprocessing video...")
        self.video_capture = cv2.VideoCapture(self.path_src)
        ret, frame = self.video_capture.read()
        count = 0
        while ret:
            self.mag_algorithm.append_frame(frame)
            ret, frame = self.video_capture.read()
            count += 1

        self.mag_algorithm._compute_differences(progress=True)
        self.mag_algorithm.apply_temporal_filter()
        self.video_capture.release()
        logger.info("Done.")
