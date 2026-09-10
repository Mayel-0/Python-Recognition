import os


os.environ["OPENCV_LOG_LEVEL"] = "FATAL"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|buffer_size;2097152|max_delay;100000|vcodec;mjpeg"
)

from ui.cli import run_threaded_rtsp


if __name__ == "__main__":
    run_threaded_rtsp()
