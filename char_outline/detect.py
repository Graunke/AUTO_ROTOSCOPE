from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import cv2
import numpy as np
from ultralytics import YOLO



def process_video(video_path: Path, model: YOLO, output_root: Path, conf: float) -> None:
    output_dir = output_root / f"{video_path.stem}_masks"
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        capture.release()
        raise RuntimeError(f"Could not determine source frame rate for: {video_path}")

    saved_frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            saved_frames += 1
            height, width = frame.shape[:2]
            output_frame = frame.copy()
            print(f"Processing frame {saved_frames}/{int(capture.get(cv2.CAP_PROP_FRAME_COUNT))} of {video_path.name}...")

            result = cast(
                Any,
                list(model.predict(
                    source=frame,
                    classes=[0],  # COCO class 0 is person
                    conf=conf,
                    retina_masks=True,
                    verbose=False,
                ))[0],
            )
            if result.masks is not None:
                person_mask = np.zeros((height, width), dtype=np.uint8)
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    if mask.shape != (height, width):
                        mask = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_NEAREST)
                    person_mask[mask > 0.5] = 255

                # Paint the person white
                output_frame[person_mask > 0] = (255, 255, 255)

                # Draw a black outline (OpenCV colors are BGR)
                contours, _ = cv2.findContours(
                person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output_frame, contours, -1, (0, 0, 0), thickness=4)

            output_path = output_dir / f"{saved_frames:06d}.png"
            if not  cv2.imwrite(str(output_path), output_frame):
                raise RuntimeError(f"Could not write mask: {output_path}")
    finally:
        capture.release()

    print(f"Saved {saved_frames} frame masks to: {output_dir}")


def run(
    videos: list[Path],
    model_path: str,
    output_dir: Path,
    conf: float,
) -> None:
    model = YOLO(model_path)
    for video in videos:
        process_video(video, model, output_dir, conf)

