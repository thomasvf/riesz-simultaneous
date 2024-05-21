import cv2
import numpy as np

from pathlib import Path
from typing import Union


def read_video(path_video: Union[str, Path]) -> np.ndarray:
    """Read video from file.

    Parameters
    ----------
    path_video : Union[str, Path]
        Path to video

    Returns
    -------
    np.ndarray
        Video in a numpy array.
    """
    path_video = Path(path_video)
    vc = cv2.VideoCapture(str(path_video))

    frames = []
    ret, frame = vc.read()
    while ret:
        frames.append(frame)
        ret, frame = vc.read()

    video = np.array(frames)
    return video


def write_video(video: np.ndarray, output_path: Union[str, Path], fr: int = 30):
    """
    Save video to file.

    Parameters
    ----------
    video : np.ndarray
        Video to save. Shape (N, H, W, C)
    output_path : Union[str, Path]
        Output path of the video
    fr : int, optional
        Frame rate, by default 30
    """
    output_path = Path(output_path).resolve()
    w, h = video.shape[2], video.shape[1]

    writer_bp = cv2.VideoWriter(
        str(output_path.resolve()), 
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fr, (w, h)
    )
    num_frames = video.shape[0]
    for k in range(num_frames):
        writer_bp.write(video[k])
    writer_bp.release()