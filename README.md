# Auto Rotoscope

Automatically detects people in video frames, applies a texture or white fill to the detected people, draws an outline, and rebuilds the frames into an MP4 video.

![Example output](assets/video.gif)

## Setup

### Requirements

- Python 3.12 (the project environment currently uses Python 3.12)
- pip
- A supported video file: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, or `.m4v`

### Install

From the project root, create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The default segmentation model path is `models/yolo26n-seg.pt`. Put the model weights there, or provide a different model path with `--model`. Model weights and input/output media are excluded from Git by this repository's `.gitignore`; do not commit large media or model files unless you intentionally change that policy.

## How to use it

### 1. Add your files

Put one or more source videos in the `video/` directory (singular). For the texture, put a `.jpg`, `.jpeg`, or `.png` image in `ref_image/`. The program uses the first supported image it finds in that folder. If it cannot find or open a texture, it fills detected people with white instead.

```text
AUTO_ROTOSCOPE/
├── main.py
├── requirements.txt
├── video/
│   └── my_video.mp4
├── ref_image/
│   └── texture.jpg
└── models/
    └── yolo26n-seg.pt
```

Create `video/`, `ref_image/`, or `models/` if needed.

### 2. Run the program

From the project root, process every supported video in `video/` with the default model and texture folder:

```bash
python main.py
```

To process one video explicitly:

```bash
python main.py "video/my_video.mp4"
```

To choose a different texture folder, model, confidence threshold, or frame-output directory:

```bash
python main.py "video/my_video.mp4" --image "my_textures" --model "models/yolo26n-seg.pt" --conf 0.35 --output "frame_output"
```

`--image` expects a **folder** containing a `.jpg`, `.jpeg`, or `.png` file, not a path to a single image. `--conf` controls the person-detection confidence threshold (default: `0.25`).

### 3. Find the results

For a source named `my_video.mp4`, the program saves processed PNG frames in:

```text
frame_output/my_video_masks/
```

It rebuilds those frames into:

```text
rotoscope_video/my_video_rotoscoped.mp4
```

The output video frame rate is currently set to 59.94 FPS in `main.py`.

### Rebuild a video from existing frames

If the PNG frames have already been generated, skip detection and rebuild the video:

```bash
python main.py "video/my_video.mp4" --rebuild-only true
```

This requires existing frames in `frame_output/my_video_masks/`.

## Examples

### Process all videos in the default input folder

```bash
python main.py
```

### Process one video and use a custom texture folder

```bash
python main.py "video/clip.mov" --image "assets/my_texture"
```

### Process one video with a different model and confidence threshold

```bash
python main.py "video/clip.mp4" --model "models/yolo26s-seg.pt" --conf 0.4
```

### Process videos from another folder

```bash
python main.py --input-vid "my_video_collection"
```

### Show command-line options

```bash
python main.py --help
```

## Notes

- Run commands from the repository root so the default relative paths resolve correctly.
- The current implementation detects COCO's `person` class and processes frames sequentially; long or high-resolution videos can take a while.
- `videos_player.py` is a separate utility that displays the first two supported videos in `videos/` side by side and saves `combined_video.mp4` in the project root.

