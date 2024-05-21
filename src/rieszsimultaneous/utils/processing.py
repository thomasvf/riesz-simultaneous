import cv2
import numpy as np


def amplitude_weighted_blur(src, amplitude, ker_size, ker_sigma):
    num = cv2.GaussianBlur(src * amplitude, (0, 0), ker_sigma)
    den = cv2.GaussianBlur(amplitude, (0, 0), ker_sigma)
    dst = num / (den + 1e-9)
    return dst


def show_mat(m):
    mat_show = cv2.normalize(m, 0, 255, cv2.NORM_MINMAX)
    mat_show = mat_show.astype(np.uint8)

    cv2.imshow("show mat", mat_show)
    cv2.waitKey(0)


def video_pyramid_size(path_src, min_dim=8):
    vc = cv2.VideoCapture(path_src)
    ret, frame = vc.read()
    h, w, c = frame.shape
    vc.release()
    return max_pyramid_size(h, w, min_dim=min_dim)


def max_pyramid_size(h, w, min_dim=8):
    """
    Compute maximum pyramid for frame with given sizes.

    :param h: height of frames
    :param w: width frames
    :param min_dim: minimum dimension of any dimension of last frame.
    :return: number of levels of laplacian pyramid (and similars), including the residue
    """
    n = 0
    hi = h
    wi = w
    while hi > min_dim and wi > min_dim:
        n += 1
        hi = hi // 2
        wi = wi // 2

    return n


def get_gaussian_pyramid(frame_src, num_levels):
    """
    Compute gaussian pyramid from frame in format (h, w)

    :param frame_src:
    :param num_levels:
    :return:
    """
    rows, cols = frame_src.shape[:2]
    if len(frame_src.shape) == 3:
        channels = frame_src.shape[2]
    else:
        channels = 0

    level = np.zeros(frame_src.shape, dtype=np.float32)
    np.copyto(level, frame_src)
    levels = [level]

    level_rows = rows
    level_cols = cols
    for i in range(0, num_levels):
        level_rows = (level_rows + 1) // 2
        level_cols = (level_cols + 1) // 2
        new_shape = (level_rows, level_cols) if not channels else (level_rows, level_cols, channels)

        new_level = np.zeros(new_shape, dtype=np.float32)
        cv2.pyrDown(level, new_level)

        levels.append(new_level)

        level = np.zeros(new_shape, dtype=np.float32)
        np.copyto(level, new_level)

    return levels
