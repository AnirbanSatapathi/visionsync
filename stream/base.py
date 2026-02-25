from abc import ABC, abstractmethod
from typing import Optional, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')

@dataclass(frozen=True)
class FrameSnapshot(Generic[T]):
    frame: Optional[T]
    frame_id: int
    monotonic_timestamp: float
    reconnect_count: int

@dataclass(frozen=True)
class StreamStats:
    uptime_seconds: float
    total_frames: int
    reconnects: int
    connected: bool

class BaseStream(ABC):
    """
    Abstract contract for video ingestion streams.

    Implementations must:
    - Perform ingestion asynchronously.
    - Guarantee get_latest_frame() is wait-free on I/O.
    - Ensure returned frames are read-only views.
    - Be safe to call from a thread different than the ingestion thread.
    """

    @abstractmethod
    def start(self) -> None:
        """Idempotent. Must not spawn duplicate ingestion threads."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Gracefully stop ingestion and release resources within bounded time."""
        pass

    @abstractmethod
    def get_latest_frame(self) -> FrameSnapshot:
        """Return the freshest available frame snapshot immediately."""
        pass

    @abstractmethod
    def stats(self) -> StreamStats:
        """Return connection-level metrics."""
        pass

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False