import os
import cv2
import numpy as np


class BandSyntheticVideo:
    """
    Represent synthetic video of circle moving periodically and whose color changes also persiodically.
    """

    def __init__(self, freq_motion=0.95, freq_color_change=0.95, sampling_rate=30, frame_rate=30,
                 w=512, h=512, c=3, length=300, spatial_freq=2.0, amplitude_cc=0.01,
                 amplitude_motion=1.0, perfect_band=False):
        """
        Initialize parameters for synthetic video

        :param freq_motion: motion frequency of circle along direction x. In Hz
        :param freq_color_change: color changes frequency of circle. In Hz
        :param sampling_rate: sampling rate of video. Hz
        :param frame_rate: frame rate for output video. Hz
        :param w: width of video
        :param h: height of video
        :param c: components
        :param length: number of frames
        :param spatial_freq: spatial frequency of band used to build the circle
        :param amplitude_cc: amplitude of color change
        :param amplitude_motion: amplitude of motion
        :param perfect_band: if False, cuts band so that only a single cycle appears.
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
        """
        Generate synthetic video for testing Riesz amplitude magnification.

        :return: numpy array with video. (t, h, w, c)
        """
        k = np.arange(0, self.length)
        f_space = self.spatial_freq / np.sqrt(self.w ** 2 + self.h ** 2)
        n = np.arange(0, self.h)
        m = np.arange(0, self.w)

        m, n = np.meshgrid(m, n)
        ds_k = self.amplitude_motion * np.sin(2 * np.pi * (self.freq_motion / self.sampling_rate) * k)
        da_k = self.amplitude_cc * np.sin(2 * np.pi * (self.freq_color_change / self.sampling_rate) * k)
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

            # print(np.min(gray), np.max(gray))
            # gray = np.clip(gray, -1, 1)
            # gray = (gray + 1)/2 * 255.0
            gray = (gray - np.min(gray))/2.0 * 255.0
            # gray = (gray +)

            # r = np.clip(r, 0, 1/(2*f_space))
            # gray = 0.25 + (1 + da) * 0.25 * np.cos(2 * np.pi * f_space * r)
            # gray = np.clip(gray, 0, 1)
            # gray *= 255.0

            im = np.zeros((self.h, self.w, self.c), dtype=np.uint8)
            im[:, :, 0] = gray.astype(np.uint8)
            im[:, :, 1] = gray.astype(np.uint8)
            im[:, :, 2] = gray.astype(np.uint8)

            video.append(im)

        return np.squeeze(np.array(video))  # check if squeeze is necessary

    def get_output_path(self, path_dir):
        """
        Return path to which generated video is saved.

        :param path_dir: path for folder in which to save the video
        :return: full path to video
        """
        filename = "synthetic_circle_fspatial{}_fcc{}_fmotion{}.mp4".format(self.spatial_freq,
                                                                            self.freq_color_change,
                                                                            self.freq_motion)
        path_dst = os.path.join(path_dir, filename)
        return path_dst

    def generate_video(self, path_dir, show=False):
        video = self.get_synthetic_video_riesz_mag()
        path_dst = self.get_output_path(path_dir)
        writer = cv2.VideoWriter(path_dst, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                 self.sampling_rate, (self.w, self.h))
        for frame in video:
            if show:
                cv2.imshow("generate_video", frame)
                cv2.waitKey(int(1000/self.frame_rate))
            writer.write(frame)
        writer.release()

        if show:
            cv2.destroyAllWindows()

        return path_dst