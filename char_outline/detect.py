from __future__ import annotations
from pathlib import Path
from typing import Any, cast
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path

def process_video(video_path: Path, model: YOLO, output_root: Path, conf: float, image: Path) -> None:
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



                folder = Path(image)
                # Ensure the folder exists (if you expect it to be pre‑populated, you might skip this)
                if not folder.is_dir():
                    print(f"Folder '{folder}' does not exist or is not a directory.")

                else:
                    # Find the first image file with a matching extension
                    image_path = None
                    for item in folder.iterdir():
                        if item.is_file() and item.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            image_path = item
                            break  # take the first match

                    if image_path is None:
                        print("Não foi possível abrir image, using white rotoscoping instead")
                        # Paint the person white
                        output_frame[person_mask > 0] = (255, 255, 255) 
                    else:
                        texture = cv2.imread(str(image_path))
                        # Paint the person white
                        output_frame[person_mask > 0] = (255, 255, 255)
                        if texture is None:
                            print("using white rotoscoping")
                        else:
                            print(f"Using texture from {image_path} for rotoscoping.")
                            texture_resized = cv2.resize(texture, (width, height))
                            output_frame[person_mask > 0] = texture_resized[person_mask > 0]
                        
                    
                        

                    


                
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
    image: Path
) -> None:
    model = YOLO(model_path)
    for video in videos:
        process_video(video, model, output_dir, conf, image)

