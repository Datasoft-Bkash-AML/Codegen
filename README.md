Process frames to detect UI elements, extract text, and annotate user flows.

Usage
`mkdir -p frames`

1. Extract frames (already done in this workspace):

   ffmpeg -i "Screen Recording 2025-09-04 at 12.12.02 PM.mov" -vf fps=1 frames/frame_%04d.png

2. Create a virtualenv and install dependencies:

   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

   Note: `detectron2` installation may be required by LayoutParser's Detectron2 backend. See https://detectron2.readthedocs.io/ for platform-specific instructions.

3. Dry-run to list frames:

   python3 process_frames.py --dry-run

4. Full run (will run LayoutParser if available, otherwise use OpenCV heuristics):

   python3 process_frames.py

5. Interactive annotation (create `output_json/flow_annotations.json`):

   python3 process_frames.py --annotate

Output

- `output_json/*.json` — per-frame JSON with detected blocks and OCR text
- `annotated/*.png` — annotated images with bounding boxes
- `output_json/flow_annotations.json` — manual flow annotations (if created)

Notes

- This is a lightweight pipeline. For production-quality UI detection, install and configure Detectron2 and a UI-specific LayoutParser model or train a custom detector.
