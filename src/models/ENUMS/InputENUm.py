"""Input type enumeration."""

from enum import Enum


class InputTypes(Enum):
    MP4 = "mp4"
    MP3 = "mp3"
    WAV = "wav"
    MKV = "mkv"
    WEBM = "webm"
    TXT = "txt"
    YOUTUBE = "youtube"  # URL-based source

    @classmethod
    def audio_formats(cls) -> list["InputTypes"]:
        return [cls.MP3, cls.WAV, cls.WEBM]

    @classmethod
    def video_formats(cls) -> list["InputTypes"]:
        return [cls.MP4, cls.MKV, cls.WEBM]
