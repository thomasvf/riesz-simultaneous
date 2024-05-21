from pathlib import Path
import cv2
from importlib_resources import files
import numpy as np

from rieszsimultaneous.algorithms.riesz import RieszMagnificationFullVideo
from rieszsimultaneous.utils.experiments import BandSyntheticVideo
from rieszsimultaneous.filters import ButterworthFullVideo
from rieszsimultaneous.utils.io import read_video, write_video


def test_riesz_full_video(tmp_path):
    path_resources = Path(files("rieszsimultaneous").joinpath("resources"))
    path_input_video = path_resources / "test_data" / "test_riesz_input.mp4"
    path_magnified_video = path_resources / "test_data" / "test_riesz_output.mp4"
    path_generated_magnified_video = tmp_path / "output_generated_mag.mp4"

    h = 64
    w = 64
    fr = 30
    syn_video = BandSyntheticVideo(
        h=h, w=w, c=3, length=300, sampling_rate=fr, frame_rate=fr
    )
    path_generated = syn_video.generate_video(tmp_path)

    syn_video_generated = read_video(path_generated)
    syn_video_true = read_video(path_input_video)
    diff = np.abs(syn_video_generated - syn_video_true)
    assert (diff < 1e-3).all()

    filter_bp = ButterworthFullVideo(30, 0.833, 1.1, 1)
    mag_alg = RieszMagnificationFullVideo(temporal_filter_motion=filter_bp)
    mag_alg.set_video(str(path_input_video))
    video_mag_bp = mag_alg.magnify_video(progress=True)
    video_mag_bp = np.array(video_mag_bp)

    print(video_mag_bp.shape)
    write_video(video_mag_bp, path_generated_magnified_video)

    mag_video_generated = read_video(path_generated_magnified_video)
    mag_video_true = read_video(path_magnified_video)
    diff = np.abs(mag_video_generated - mag_video_true)
    assert (diff < 1e-3).all()



if __name__ == "__main__":
    tmp_path = Path('tmp')
    tmp_path.mkdir(exist_ok=True)
    test_riesz_full_video(tmp_path)
