import os
import time
import threading

import cv2


def open_stream(source) -> cv2.VideoCapture:
    """Ouvre une webcam ou un flux RTSP avec une latence réduite."""
    if isinstance(source, str):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
            "rtsp_transport;tcp|buffer_size;1024000|max_delay;500000"
        )
        capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        capture = cv2.VideoCapture(source)

    if not capture.isOpened():
        print(f"Impossible d'ouvrir la source : {source}")
        raise SystemExit(1)
    return capture


class RTSPStreamThread:
    """Capture un flux RTSP dans un thread pour limiter le retard du buffer."""

    def __init__(self, url: str):
        self.url = url
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = False
        self.lock = threading.Lock()
        self.thread = None
        self._connect()

    def _connect(self) -> None:
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self) -> bool:
        if not self.cap.isOpened():
            print(f"Impossible d'ouvrir le flux RTSP : {self.url}")
            return False
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return True

    def _update(self) -> None:
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.5)
                self._connect()
                continue

            ret, frame = self.cap.read()
            if ret and frame is not None and frame.size > 0:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self) -> None:
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
