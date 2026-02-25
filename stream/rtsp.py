import cv2
import threading
import time
import os
import numpy as np
from typing import Optional

from base import BaseStream, FrameSnapshot

class RTSPStream(BaseStream):
    def __init__(self, url: str, reconnect: bool = True):
        self.url = url
        self.should_reconnect = reconnect
        
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock() 
        
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._cap: Optional[cv2.VideoCapture] = None 
        
        self._timestamp: float = 0.0
        self._start_time: float = 0.0
        self._total_frames: int = 0
        self._reconnect_count: int = 0 
        self._connected: bool = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return 
            
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True, name="RTSPStreamReader")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

        if self._cap:
            self._cap.release()
            self._cap = None

        self._connected = False

    def get_latest_frame(self) -> FrameSnapshot[np.ndarray]:
        with self._lock:
            frame_ref = self._latest_frame
            frame_id = self._frame_id
            ts = self._timestamp
            reconnects = self._reconnect_count

        if frame_ref is None:
            return FrameSnapshot(
                frame=None,
                frame_id=frame_id,
                monotonic_timestamp=ts,
                reconnect_count=reconnects
            )
            
        view = frame_ref.view()
        view.flags.writeable = False

        return FrameSnapshot(
            frame=view,
            frame_id=frame_id,
            monotonic_timestamp=ts,
            reconnect_count=reconnects,
        )

    @staticmethod
    def _configure_ffmpeg_env() -> None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "fflags;nobuffer|flags;low_delay|analyzeduration;0|probesize;32"

    def _reader_loop(self) -> None:
        backoff = 1.0
        max_backoff = 16.0
        
        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                self._connected = False

                try:
                    self._configure_ffmpeg_env()
                    cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                    if not cap.isOpened():
                        raise ConnectionError("Failed to open RTSP stream.")
                        
                    self._cap = cap
                    self._connected = True
                    
                    if self._total_frames > 0:
                        with self._lock:
                            self._reconnect_count += 1
                    backoff = 1.0

                except Exception:
                    if not self.should_reconnect:
                        break
                    time.sleep(backoff)
                    backoff = min(backoff * 2, max_backoff)
                    continue

            ret, frame = self._cap.read()
            
            if not ret or frame is None:
                self._cap.release()
                self._cap = None
                self._connected = False

                if not self.should_reconnect: 
                    break

                time.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
                continue
                
            now = time.perf_counter()

            with self._lock:
                self._latest_frame = frame
                self._frame_id += 1
                self._timestamp = now
                self._total_frames += 1
        
        self._connected = False