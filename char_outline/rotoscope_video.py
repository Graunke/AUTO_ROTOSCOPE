from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import cv2


def rebuild_video(
    frames_dir: Path,
    output: Path,
    fps: float | None = None,
    source_video: Path | None = None,
) -> None:
    frame_files = sorted(frames_dir.glob("*.png"))
    if not frame_files:
        raise RuntimeError(f"No PNG frames found in: {frames_dir}")

    if fps is None:
        if source_video is None:
            raise RuntimeError("A source video or FPS is required")
        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise RuntimeError(f"Could not open source video: {source_video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()

    if fps is None or fps <= 0:
        raise RuntimeError(f"Invalid output FPS: {fps}")

    rebuilt_video = cv2.VideoCapture(f"{frames_dir}/%06d.png")
    rebuilt_video.set(cv2.CAP_PROP_FPS, fps)
    width = int(rebuilt_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(rebuilt_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cast(Any, getattr(cv2, "VideoWriter_fourcc"))(*"mp4v")
    total_frames = int(rebuilt_video.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))

    counter = 0
    while rebuilt_video.isOpened():
        ret, frame = rebuilt_video.read()
        if not ret:
            break
        print(f"Processing frame {counter}")
        writer.write(frame)  # Write frame to file
        counter += 1

    rebuilt_video.release()
    writer.release()
    print(f"Saved {total_frames} frames to: {output}")
