import argparse
from pathlib import Path
from typing import Any, cast

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a video from numbered PNG frames.")
    parser.add_argument("--frames", type=Path, default=Path("frame_output/person_masks"), help="Folder containing 000001.png, 000002.png, ...")
    parser.add_argument("--output", type=Path, default=Path("rotoscope_video/output.mp4"), help="Output MP4 path")
    parser.add_argument("--fps", type=float, help="Output frames per second")
    parser.add_argument("--source-video", type=Path, default=Path("video/person.mp4"), help="Use this video's FPS when --fps is omitted")
    args = parser.parse_args()

    frames_dir = Path(args.frames)
    output = Path(args.output)
    source_video = Path(args.source_video) if args.source_video else None

    # Definir a taxa de quadros (FPS - Frames Por Segundo), por exemplo, 30
    if args.fps is not None:
        fps = args.fps
    elif source_video:
        capture = cv2.VideoCapture(str(source_video))
        if not capture.isOpened():
            raise SystemExit(f"Could not open source video: {source_video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        capture.release()
        if fps <= 0:
            raise SystemExit(f"Could not determine FPS from source video: {source_video}")
    else:
        raise SystemExit("Provide --fps or --source-video to determine the output frame rate")

    images = sorted(frames_dir.glob("*.png"))

    if not images:
        raise SystemExit(f"No frames found in: {frames_dir}")

    frame = cv2.imread(str(images[0]))
    if frame is None:
        raise SystemExit(f"Could not read first frame: {images[0]}")
    height, width = frame.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cast(Any, getattr(cv2, "VideoWriter_fourcc"))(*"mp4v")
    video = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not video.isOpened():
        raise SystemExit(f"Could not create output video: {output}")

    try:
        for image in images:
            frame = cv2.imread(str(image))
            if frame is None:
                raise SystemExit(f"Could not read frame: {image}")
            video.write(frame)
    finally:
        video.release()
    print(f"Video saved successfully: {output}")

if __name__ == "__main__":
    main()