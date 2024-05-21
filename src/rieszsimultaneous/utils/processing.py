from pathlib import Path
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


def video_pyramid_size(path_src: str, min_dim=8):
    path_src = str(Path(path_src).resolve())
    vc = cv2.VideoCapture(path_src)
    ret, frame = vc.read()
    h, w, c = frame.shape
    vc.release()
    return max_pyramid_size(h, w, min_dim=min_dim)


def max_pyramid_size(h: int, w: int, min_dim: int = 8) -> int:
    """Compute maximum size of a pyramid for a frame with  thegiven sizes.

    Parameters
    ----------
    h : int
        Height of the frame
    w : int
        Width of the frame
    min_dim : int, optional
        Minimum size of any dimension in the last frame, by default 8

    Returns
    -------
    int
        Number of levels of laplacian pyramid (and similars), including the residue
    """
    n = 0
    hi = h
    wi = w
    while hi > min_dim and wi > min_dim:
        n += 1
        hi = hi // 2
        wi = wi // 2

    return n


def get_gaussian_pyramid(frame_src: np.ndarray, num_levels: int):
    """Compute gaussian pyramid for frame_src with num_levels levels.

    Parameters
    ----------
    frame_src : np.ndarray
        Frame to compute the pyramid. Shape (H, W, C)
    num_levels : int
        Number of levels of the pyramid, besides the original image (which is the
        level 0)

    Returns
    -------
    Sequence[np.ndarray]
        List of levels of the pyramid. The first element is the original image.
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
