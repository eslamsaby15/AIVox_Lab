"""Data models for speaker diarization results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Segment:
    """A single speaker turn."""
    speaker: str
    text: str
    timestamp: Optional[float] = None  # optional start-time in seconds

    def __str__(self) -> str:
        ts = f"[{self.timestamp:.1f}s] " if self.timestamp is not None else ""
        return f"{ts}{self.speaker}: {self.text}"


@dataclass
class DiarizationResult:
    """Full diarization output containing all speaker segments."""
    segments: List[Segment] = field(default_factory=list)

    @property
    def total_speakers(self) -> int:
        """Number of unique speakers detected."""
        return len({seg.speaker for seg in self.segments})

    @property
    def transcript(self) -> str:
        """Full transcript as plain text."""
        return "\n".join(str(seg) for seg in self.segments)