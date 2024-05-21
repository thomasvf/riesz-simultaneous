from rieszsimultaneous.gui import MagnificationManager


def main():
    path_src = "/home/thomas/work/tcc/videos/baby2.mp4"
    gui_manager = MagnificationManager(30, 150, (140.0/60, 160.0/60), (140.0/60, 160.0/60))
    gui_manager.start_gui_video(path_src)
