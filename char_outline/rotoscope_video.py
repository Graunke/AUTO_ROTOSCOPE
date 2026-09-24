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

    first_frame = cv2.imread(str(frame_files[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read frame: {frame_files[0]}")
    height, width = first_frame.shape[:2]

    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cast(Any, getattr(cv2, "VideoWriter_fourcc"))(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output}")

    try:
        for frame_file in frame_files:
            frame = cv2.imread(str(frame_file))
            if frame is None:
                raise RuntimeError(f"Could not read frame: {frame_file}")
            if frame.shape[:2] != (height, width):
                raise RuntimeError(f"Frame size differs from first frame: {frame_file}")
            writer.write(frame)
    finally:
        writer.release()

    print(f"Saved {len(frame_files)} frames to: {output}")
