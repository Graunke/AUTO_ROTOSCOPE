import argparse
from pathlib import Path
from char_outline import detect
from char_outline import rotoscope_video


VIDEO_DIR = "video"
OUTPUT_DIR = "frame_output"
VIDEO_OUTPUT_DIR = "rotoscope_video"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create black-and-white person masks for every frame in a video."
    )
    parser.add_argument(
        "video", nargs="?", type=Path,
        help="Video file to process (defaults to every supported video in ./video).",
    )
    parser.add_argument("--model", default="models/yolo26n-seg.pt", help="Ultralytics segmentation model.")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR, help="Output directory.")
    parser.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold.")
    parser.add_argument("--input-vid", type=Path, default=VIDEO_DIR, help="Input video directory.")
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_args()
    videos = [args.video] if args.video else sorted(
        path for path in args.input_vid.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ) if args.input_vid.exists() else []

    if not videos:
        raise SystemExit(
            f"No supported video files found in {args.input_vid}; add one or pass a video path."
        )
    missing = [path for path in videos if not path.is_file()]
    if missing:
        raise SystemExit("Video file not found: " + ", ".join(map(str, missing)))

    detect.run(videos, args.model, args.output, args.conf)

    for video in videos:
        frames_dir = args.output / f"{video.stem}_masks"
        output_video = Path(VIDEO_OUTPUT_DIR) / f"{video.stem}_rotoscoped.mp4"
        rotoscope_video.rebuild_video(
            frames_dir=frames_dir,
            output=output_video,
            fps=None,
            source_video=video,
        )

if __name__ == "__main__":
    main()