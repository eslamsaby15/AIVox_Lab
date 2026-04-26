"""YouTube audio downloader and file upload handler."""

from __future__ import annotations

import os

import librosa
import soundfile as sf
import yt_dlp

from .BaseController import BaseController
from .ProjectController import ProjectController


class Youtube(BaseController):
    def __init__(self) -> None:
        super().__init__()
        self.project_controller = ProjectController()

    def download(self, url: str) -> str:
        """
        Download audio from a YouTube URL, convert to WAV, and return the path.
        """
        video_id = url.rstrip("/").split("/")[-1].split("?")[0]
        project_path = self.project_controller.get_project_path(video_id)
        output_template = os.path.join(project_path, "%(id)s.%(ext)s")

        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": output_template,
            "quiet": True,
            "noplaylist": True,
            "postprocessors": [{
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }],
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # yt-dlp renames the file after postprocessing
                wav_path = os.path.join(project_path, f"{info['id']}.wav")
        except yt_dlp.utils.DownloadError as exc:
            raise RuntimeError(f"Unable to download video: {exc}") from exc

        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Expected WAV file not found: {wav_path}")

        return wav_path

    # Backward-compat alias
    def Download(self, url: str) -> str:  # noqa: N802
        return self.download(url)

    def save_uploaded_file(self, upload_file) -> str:
        """Save a Streamlit UploadedFile to disk and return the path."""
        project_key = self.generate_random_string()
        project_path = self.project_controller.get_project_path(upload_file.name)

        filename = f"{project_key}_{upload_file.name}"
        output_path = os.path.join(project_path, filename)

        with open(output_path, "wb") as fh:
            fh.write(upload_file.getbuffer())

        return output_path

    # Backward-compat alias
    def save_dir(self, upload_file) -> str:
        return self.save_uploaded_file(upload_file)