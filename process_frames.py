import os
import glob
import json
import argparse
from typing import List, Dict, Any, Tuple
# teserract path
teserract_cmd_path = r'/usr/bin/tesseract'  # Update this path as needed

# Configuration
FRAME_DIR = 'frames'
OUTPUT_DIR = 'output_json'
ANNOTATED_DIR = 'annotated'
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ANNOTATED_DIR, exist_ok=True)


def lazy_imports():
    """Lazy import heavy libraries and return them. This lets --dry-run run fast."""
    import cv2
    import pytesseract
    # LayoutParser import can fail if detectron2 not installed; handle gracefully
    try:
        import layoutparser as lp
    except Exception:
        lp = None
    # try easyocr as a fallback (may not be installed)
    try:
        import easyocr
    except Exception:
        easyocr = None
    return cv2, pytesseract, lp, easyocr
    


def detect_with_layoutparser(lp, image) -> List[Dict[str, Any]]:
    """Run LayoutParser detection if available. Returns list of blocks.
    Blocks: {'type','score','bbox'}
    """
    model = None
    # Use a light PubLayNet model URI; if lp can't fetch model offline, caller should handle
    try:
        model = lp.Detectron2LayoutModel(
            'lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
            extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.4],
        )
    except Exception:
        model = None

    if model is None:
        return []

    layout = model.detect(image)
    blocks = []
    for b in layout:
        x1, y1, x2, y2 = map(int, b.coordinates)
        blocks.append({'type': b.type, 'score': float(b.score), 'bbox': [x1, y1, x2, y2]})
    return blocks


def detect_with_opencv(cv2, image) -> List[Dict[str, Any]]:
    """Simple OpenCV-based detection to find rectangular UI-like elements using edge + contour heuristics.
    Returns list of blocks with 'bbox' and 'score' (heuristic).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    blocks = []
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        # Heuristic filters: ignore very small or extremely large boxes
        if cw < 30 or ch < 20:
            continue
        if cw > 0.9 * w and ch > 0.9 * h:
            continue
        # Approximate rectangularity
        area = cw * ch
        rect_area = cv2.contourArea(cnt)
        score = min(1.0, rect_area / (area + 1e-6))
        blocks.append({'type': 'rect', 'score': float(score), 'bbox': [x, y, x + cw, y + ch]})
    # Merge overlapping boxes (simple NMS-like)
    blocks = _merge_overlapping_boxes(blocks)
    return blocks


def _merge_overlapping_boxes(blocks: List[Dict[str, Any]], iou_thresh=0.3) -> List[Dict[str, Any]]:
    # Simple greedy merge by area
    if not blocks:
        return []
    boxes = [b['bbox'] for b in blocks]
    scores = [b['score'] for b in blocks]
    inds = list(range(len(boxes)))
    keep = []
    while inds:
        i = max(inds, key=lambda k: scores[k])
        keep.append(blocks[i])
        inds.remove(i)
        bx = boxes[i]
        rem = []
        for j in inds:
            if _iou(bx, boxes[j]) < iou_thresh:
                rem.append(j)
        inds = rem
    return keep


def _iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter + 1e-9)


def ocr_on_blocks(pytesseract, easyocr, image, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for b in blocks:
        x1, y1, x2, y2 = map(int, b['bbox'])
        crop = image[y1:y2, x1:x2]
        r = dict(b)
        try:
            text = pytesseract.image_to_string(crop, lang='eng')
            r['text'] = text.strip()
        except Exception as e:
            # Tesseract binary might be missing or another OCR error occurred
            r['text'] = ''
            r['ocr_error'] = str(e)
        results.append(r)
    return results


def easyocr_read(easyocr, image) -> str:
    """Use EasyOCR reader to extract text from an image. Returns combined text."""
    if easyocr is None:
        return ''
    # Create a cached reader on module if not present
    if not hasattr(easyocr, '_cached_reader'):
        try:
            # English only for speed
            easyocr._cached_reader = easyocr.Reader(['en'], gpu=False)
        except Exception:
            return ''
    try:
        res = easyocr._cached_reader.readtext(image)
        texts = [r[1] for r in res]
        return ' '.join(texts)
    except Exception:
        return ''


def annotate_image(cv2, image, blocks: List[Dict[str, Any]]) -> Any:
    vis = image.copy()
    for b in blocks:
        x1, y1, x2, y2 = map(int, b['bbox'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = b.get('type', '') + (f" {b.get('text','')[:30]}" if b.get('text') else '')
        cv2.putText(vis, label, (x1, max(y1 - 6, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return vis


def process_frame(frame_path: str, use_lp=True) -> Dict[str, Any]:
    cv2, pytesseract, lp, easyocr = lazy_imports()
    image = cv2.imread(frame_path)
    if image is None:
        raise ValueError(f'Could not read image {frame_path}')

    # 1) Try LayoutParser
    blocks = []
    if use_lp and lp is not None:
        try:
            blocks = detect_with_layoutparser(lp, image)
        except Exception:
            blocks = []

    # 2) Fallback to OpenCV if no blocks found
    if not blocks:
        blocks = detect_with_opencv(cv2, image)

    # 3) OCR on blocks (try pytesseract, record error; we'll fallback to EasyOCR later if needed)
    ocr_results = ocr_on_blocks(pytesseract, easyocr, image, blocks)

    # 4) Global OCR (whole frame) as well: try pytesseract first, then easyocr
    full_text = ''
    try:
        full_text = pytesseract.image_to_string(image, lang='eng')
    except Exception:
        full_text = ''
    if (not full_text or full_text.strip() == '') and easyocr is not None:
        try:
            full_text = easyocr_read(easyocr, image)
        except Exception:
            full_text = ''


    return {
        'frame': os.path.basename(frame_path),
        'blocks': ocr_results,
        'full_text': full_text.strip(),
        'shape': [int(x) for x in image.shape[:2]]
    }


def dry_run():
    files = sorted(glob.glob(os.path.join(FRAME_DIR, '*.png')))
    print(f'Found {len(files)} frames in {FRAME_DIR}')
    for f in files[:5]:
        print(' ', os.path.basename(f))


def save_results(frame_result: Dict[str, Any], annotated_image=None):
    name = frame_result['frame']
    base = os.path.splitext(name)[0]
    out_json = os.path.join(OUTPUT_DIR, base + '.json')
    with open(out_json, 'w') as f:
        json.dump(frame_result, f, indent=2)
    if annotated_image is not None:
        import cv2
        cv2.imwrite(os.path.join(ANNOTATED_DIR, base + '.png'), annotated_image)


def interactive_annotate():
    """Simple CLI to step through frames and collect manual annotations for flows."""
    files = sorted(glob.glob(os.path.join(FRAME_DIR, '*.png')))
    annotations = []
    for idx, fp in enumerate(files):
        print(f'[{idx}] {os.path.basename(fp)}')
    print('\nEnter comma-separated frame indices for a flow (e.g. 0,1,2) or blank to exit:')
    s = input().strip()
    if not s:
        print('No annotation provided')
        return
    try:
        indices = [int(x.strip()) for x in s.split(',')]
    except Exception:
        print('Invalid input')
        return
    flow = [os.path.basename(sorted(glob.glob(os.path.join(FRAME_DIR, '*.png')))[i]) for i in indices]
    annotations.append({'flow': flow})
    with open(os.path.join(OUTPUT_DIR, 'flow_annotations.json'), 'w') as f:
        json.dump(annotations, f, indent=2)
    print('Saved flow_annotations.json')


def run_all(dry=False, annotate=False, use_lp=True, save_annotated=True):
    if dry:
        dry_run()
        return
    cv2, pytesseract, lp, easyocr = lazy_imports()
    files = sorted(glob.glob(os.path.join(FRAME_DIR, '*.png')))
    for fp in files:
        print('Processing', fp)
        res = process_frame(fp, use_lp=use_lp)
        # create annotated image
        try:
            vis = annotate_image(cv2, cv2.imread(fp), res['blocks'])
        except Exception:
            vis = None
        save_results(res, annotated_image=vis if save_annotated else None)

    if annotate:
        interactive_annotate()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--annotate', action='store_true', help='Open interactive annotator to create flows')
    p.add_argument('--no-layoutparser', dest='use_lp', action='store_false')
    p.add_argument('--no-annotated', dest='save_annotated', action='store_false')
    return p.parse_args()


def main():
    args = parse_args()
    run_all(dry=args.dry_run, annotate=args.annotate, use_lp=args.use_lp, save_annotated=args.save_annotated)


if __name__ == '__main__':
    main()
