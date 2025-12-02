#!/usr/bin/env python3
# plate_two_line_read.py
# Usage: python plate_two_line_read.py /path/to/image.jpg

import sys, re, os
from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
import json

# ---------- CONFIG ----------
YOLO_MODEL = "last.pt"   # <-- set to your best.pt (or last.pt)
OUT_DIR = "output"
CONF_THRESHOLD = 0.25
OCR_LANGS = ['en']
OCR_MIN_HEIGHT = 12
OCR_RESIZE_HEIGHT = 128

os.makedirs(OUT_DIR, exist_ok=True)

# ---------- helpers ----------
reader = easyocr.Reader(OCR_LANGS, gpu=False)

def preprocess_gray_for_split(bgr):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    # equalize + blur to reduce noise
    g = cv2.equalizeHist(g)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return g

def binarize_for_hist(gray):
    # adaptive threshold or Otsu
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ensure text is white on black background for histogram
    if np.mean(th) > 127:
        th = 255 - th
    return th

def find_horizontal_split(bin_img):
    # bin_img: binary image (0 or 255)
    # compute white pixel count per row
    hist = np.sum(bin_img == 255, axis=1).astype(np.float32)

    # reshape to 2D so OpenCV can blur
    hist_2d = hist.reshape(-1, 1)

    # smooth with vertical kernel (21×1)
    hist_s = cv2.blur(hist_2d, (1, 21)).flatten()

    # find 2 largest peaks (rows with most white pixels)
    if len(hist_s) < 2:
        return bin_img.shape[0] // 2

    peaks = np.argsort(hist_s)[-2:]  # 2 largest values
    top_peak, bottom_peak = np.sort(peaks)

    # find minimum between peaks
    if bottom_peak - top_peak <= 4:
        split = (top_peak + bottom_peak) // 2
    else:
        segment = hist_s[top_peak:bottom_peak+1]
        min_idx = np.argmin(segment)
        split = top_peak + min_idx

    # clamp split inside image
    H = bin_img.shape[0]
    split = max(1, min(split, H - 2))

    return split

def ocr_read_image(img_bgr) -> Tuple[str, float]:
    """
    Run EasyOCR (with some preprocessing) and return best text + confidence
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # upscale small images
    h,w = gray.shape[:2]
    if h < OCR_MIN_HEIGHT and h>0:
        scale = OCR_RESIZE_HEIGHT / h
        img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # variants
    imgs = []
    # variant original RGB
    imgs.append(img_bgr[:,:,::-1])  # BGR->RGB for easyocr
    # variant grayscale with adaptive threshold
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    g = cv2.bilateralFilter(g, 5, 75, 75)
    try:
        th = cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        if np.mean(th) > 127:
            th = 255 - th
        imgs.append(cv2.cvtColor(th, cv2.COLOR_GRAY2RGB))
    except Exception:
        pass
    best_txt = ""
    best_conf = 0.0
    for variant in imgs:
        try:
            res = reader.readtext(variant, detail=1)
        except Exception:
            res = []
        for bbox, text, conf in res:
            txt = re.sub(r'[^A-Za-z0-9\- ]+', '', text).strip().upper()
            if len(txt) < 1:
                continue
            # boost if pattern-like
            # compute simple score
            score = float(conf)
            if re.search(r'[0-9]', txt) and re.search(r'[A-Z]', txt):
                score += 0.1
            if score > best_conf:
                best_conf = score
                best_txt = txt
    return best_txt, best_conf

def normalize_top_bottom(top: str, bottom: str) -> str:
    # clean
    top_c = re.sub(r'[^A-Z0-9\- ]', '', (top or '').upper()).replace(' ', '')
    bottom_c = re.sub(r'[^0-9]', '', (bottom or '').upper())
    # if top missing dash (like '74F6'), insert dash after 2 chars if plausible
    if top_c and '-' not in top_c and len(top_c) >= 3:
        top_c = top_c[:2] + '-' + top_c[2:]
    # format with spaces around dash
    top_c = top_c.replace('-', ' - ')
    # final assemble
    if top_c and bottom_c:
        final = f"{top_c} {bottom_c}"
    elif top_c:
        final = top_c
    elif bottom_c:
        final = bottom_c
    else:
        final = ""
    # collapse multiple spaces
    final = re.sub(r'\s+', ' ', final).strip()
    return final

# ---------- main pipeline ----------
def detect_and_read_2line(image_path: str,
                          model_path: str = YOLO_MODEL,
                          out_dir: str = OUT_DIR,
                          conf_thresh: float = CONF_THRESHOLD):
    model = YOLO(model_path)
    img_p = Path(image_path)
    if not img_p.exists():
        raise FileNotFoundError(image_path + " not found")
    img = cv2.imread(str(img_p))
    if img is None:
        raise RuntimeError("Cannot read image")

    # detect (single image)
    r = model.predict(source=str(img_p), imgsz=640, conf=conf_thresh, verbose=False)[0]
    annotated = img.copy()
    plates = []

    if hasattr(r, "boxes") and len(r.boxes) > 0:
        for box, conf in zip(r.boxes.xyxy, r.boxes.conf):
            x1,y1,x2,y2 = map(int, box.tolist())
            # expand margin more vertically to include both lines
            h = y2 - y1; w = x2 - x1
            pad_top = int(0.20 * h)
            pad_bot = int(0.20 * h)
            pad_lr = int(0.08 * w)
            xa = max(0, x1 - pad_lr)
            ya = max(0, y1 - pad_top)
            xb = min(img.shape[1], x2 + pad_lr)
            yb = min(img.shape[0], y2 + pad_bot)
            crop = img[ya:yb, xa:xb].copy()

            # preprocess and binarize to find split
            gray = preprocess_gray_for_split(crop)
            binimg = binarize_for_hist(gray)
            split_row = find_horizontal_split(binimg)
            # if split too close to edges, fallback to center split
            if split_row < int(0.15 * crop.shape[0]) or split_row > int(0.85 * crop.shape[0]):
                split_row = crop.shape[0] // 2

            top_crop = crop[:split_row, :]
            bot_crop = crop[split_row:, :]

            top_text, top_conf = ocr_read_image(top_crop)
            bot_text, bot_conf = ocr_read_image(bot_crop)

            final_plate = normalize_top_bottom(top_text, bot_text)

            plates.append({
                "box": (xa,ya,xb,yb),
                "det_conf": float(conf),
                "top_text": top_text,
                "top_conf": float(top_conf),
                "bottom_text": bot_text,
                "bottom_conf": float(bot_conf),
                "plate": final_plate
            })

            # annotate
            cv2.rectangle(annotated, (xa,ya), (xb,yb), (0,255,0), 2)
            cv2.putText(annotated, final_plate or "UNKNOWN", (xa, max(ya-6,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    else:
        # no detections: optionally run OCR on whole image (attempt)
        top_text, top_conf = ocr_read_image(img)
        plates.append({"box": None, "plate": normalize_top_bottom(top_text,"")})

    # save outputs
    stem = img_p.stem
    out_annot = Path(out_dir) / f"{stem}_annot.jpg"
    out_txt = Path(out_dir) / f"{stem}_plate.txt"
    out_json = Path(out_dir) / f"{stem}_summary.json"
    cv2.imwrite(str(out_annot), annotated)

    with open(out_txt, "w", encoding="utf-8") as f:
        if plates:
            for p in plates:
                f.write((p.get("plate") or "NOT_FOUND") + "\n")
        else:
            f.write("NOT_FOUND\n")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"image": str(img_p), "plates": plates}, f, ensure_ascii=False, indent=2)

    print("Saved:", out_annot, out_txt, out_json)
    return plates

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plate_two_line_read.py /path/to/image.jpg")
        sys.exit(1)
    imgpath = sys.argv[1]
    if not Path(YOLO_MODEL).exists():
        print("YOLO model not found at", YOLO_MODEL, "- please set YOLO_MODEL to .pt path")
        sys.exit(2)
    os.makedirs(OUT_DIR, exist_ok=True)
    res = detect_and_read_2line(imgpath, model_path=YOLO_MODEL, out_dir=OUT_DIR, conf_thresh=CONF_THRESHOLD)
    print("Result:", res)
