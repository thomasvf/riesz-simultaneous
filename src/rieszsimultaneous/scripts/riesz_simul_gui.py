from pathlib import Path
import argparse


from rieszsimultaneous.gui import MagnificationManager


def main():
    parser = argparse.ArgumentParser(description="Riesz Simultaneous CLI")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to the video file (default: %(default)s)",
    )
    parser.add_argument(
        "--band-motion",
        type=float,
        nargs=2,
        default=(40.0, 80.0),
        help="Band of the motion magnification filter in BPM",
    )
    parser.add_argument(
        "--band-color",
        type=float,
        nargs=2,
        default=(60.0, 110.0),
        help="Band of the color variation magnification filter in BPM",
    )
    args = parser.parse_args()

    if Path(args.video).exists() is False:
        print("The video file was not found.")
        return

    lc_band_motion, hc_band_motion = args.band_motion
    lc_band_colorchange, hc_band_colorchange = args.band_color
    gui_manager = MagnificationManager(
        alpha_motion=30,
        alpha_color=150,
        band_motion=(lc_band_motion / 60, hc_band_motion / 60),
        band_color=(lc_band_colorchange / 60, hc_band_colorchange / 60),
    )
    gui_manager.start_gui_video(args.video)


if __name__ == "__main__":
    main()
