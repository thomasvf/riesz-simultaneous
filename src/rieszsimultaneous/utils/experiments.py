import os
import cv2
import numpy as np


class BandSyntheticVideo:
    """
    Represent synthetic video of circle moving periodically and whose color changes also persiodically.
    """

    def __init__(
        self,
        freq_motion: float = 0.95,
        freq_color_change: float = 0.95,
        sampling_rate: float = 30,
        frame_rate: float = 30,
        w: int = 512,
        h: int = 512,
        c: int = 3,
        length: int = 300,
        spatial_freq: bool = 2.0,
        amplitude_cc: bool = 0.01,
        amplitude_motion: bool = 1.0,
        perfect_band: bool = False,
    ):
        """Initialize object with parameters for synthetic video generation.

        Parameters
        ----------
        freq_motion : float, optional
            motion frequency of the circle along direction x (in Hz), by default 0.95
        freq_color_change : float, optional
            color changes frequency of the circle (in Hz), by default 0.95
        sampling_rate : float, optional
            sampling rate of the video (in Hz), by default 30
        frame_rate : float, optional
            Frame rate for the output video, by default 30
        w : int, optional
            Width of the video frames, by default 512
        h : int, optional
            Height of the video frames, by default 512
        c : int, optional
            Number of components of the videof rames, by default 3
        length : int, optional
            Number of frames in the videdo, by default 300
        spatial_freq : bool, optional
            Spatial frequency of the band used to build the circle, by default 2.0
        amplitude_cc : bool, optional
            Amplitude of the color change in the video, by default 0.01
        amplitude_motion : bool, optional
            Amplitude of the motion of the circle in the video, by default 1.0
        perfect_band : bool, optional
            Whether to use a perfect sinusoid or to clip it, by default False.
        """
        self.freq_motion = freq_motion
        self.freq_color_change = freq_color_change
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.w = w
        self.h = h
        self.c = c
        self.length = length
        self.spatial_freq = spatial_freq
        self.amplitude_cc = amplitude_cc
        self.amplitude_motion = amplitude_motion
        self.perfect_band = perfect_band

    def get_synthetic_video_riesz_mag(self):
        """Generate synthetic video for testing Riesz amplitude magnification

        Returns
        -------
        np.ndarray
            numpy array of shape (t, h, w, c) containing the video
        """
        k = np.arange(0, self.length)
        f_space = self.spatial_freq / np.sqrt(self.w**2 + self.h**2)
        n = np.arange(0, self.h)
        m = np.arange(0, self.w)

        m, n = np.meshgrid(m, n)
        ds_k = self.amplitude_motion * np.sin(
            2 * np.pi * (self.freq_motion / self.sampling_rate) * k
        )
        da_k = self.amplitude_cc * np.sin(
            2 * np.pi * (self.freq_color_change / self.sampling_rate) * k
        )
        peak_value = 0.8
        video = []
        for k_ in k:
            ds = ds_k[k_]
            da = da_k[k_]
            r = np.sqrt((m - ds - self.w / 2) ** 2 + (n - self.h / 2) ** 2)
            if self.perfect_band:
                gray = (1 + da) * peak_value * np.cos(2 * np.pi * f_space * r)
            else:
                r = np.clip(r, 0, 1 / (2 * f_space))  # just one peak
                gray = (1 + da) * peak_value * np.cos(2 * np.pi * f_space * r)

            gray = (gray - np.min(gray)) / 2.0 * 255.0

            im = np.zeros((self.h, self.w, self.c), dtype=np.uint8)
            im[:, :, 0] = gray.astype(np.uint8)
            im[:, :, 1] = gray.astype(np.uint8)
            im[:, :, 2] = gray.astype(np.uint8)

            video.append(im)

        return np.squeeze(np.array(video))  # check if squeeze is necessary

    def get_output_path(self, path_dir: str):
        """Return the path of the output video

        Parameters
        ----------
        path_dir : str
            Folder where the video was saved

        Returns
        -------
        str
            Full output path with the filename
        """
        filename = "synthetic_circle_fspatial{}_fcc{}_fmotion{}.mp4".format(
            self.spatial_freq, self.freq_color_change, self.freq_motion
        )
        path_dst = os.path.join(path_dir, filename)
        return path_dst

    def generate_video(self, path_dir: str, show: bool = False):
        """Generate and save video

        Parameters
        ----------
        path_dir : str
            Path with directory where to save the video
        show : bool, optional
            Show video using opencv GUI, by default False

        Returns
        -------
        str
            Path to video file
        """
        video = self.get_synthetic_video_riesz_mag()
        path_dst = self.get_output_path(path_dir)
        writer = cv2.VideoWriter(
            path_dst,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            self.sampling_rate,
            (self.w, self.h),
        )
        for frame in video:
            if show:
                cv2.imshow("generate_video", frame)
                cv2.waitKey(int(1000 / self.frame_rate))
            writer.write(frame)
        writer.release()

        if show:
            cv2.destroyAllWindows()

        return path_dst
