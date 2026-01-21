# make_background_tiles.py
import cv2, random
from pathlib import Path

img_dir = Path("data/images/train")
lbl_dir = Path("data/labels/train")
out_dir = Path("data/images/train")
out_lbl = Path("data/labels/train")
out_dir.mkdir(parents=True, exist_ok=True)
out_lbl.mkdir(parents=True, exist_ok=True)

def load_boxes(lbl_path, w, h):
    boxes = []
    if lbl_path.exists():
        for line in lbl_path.read_text().splitlines():
            cls, xc, yc, bw, bh = map(float, line.split())
            x = (xc - bw/2) * w
            y = (yc - bh/2) * h
            ww = bw * w
            hh = bh * h
            boxes.append((x, y, x+ww, y+hh))
    return boxes

def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB-xA) * max(0, yB-yA)
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / max(1e-6, areaA + areaB - inter)

for img_path in img_dir.rglob("*"):
    if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        continue
    im = cv2.imread(str(img_path))
    if im is None:
        continue
    h, w = im.shape[:2]
    lbl_path = lbl_dir / img_path.relative_to(img_dir).with_suffix(".txt")
    gts = load_boxes(lbl_path, w, h)
    for k in range(5):  # five tiles per image
        tw, th = int(0.4*w), int(0.4*h)  # tile size ratio
        for _ in range(20):
            x0 = random.randint(0, max(1, w - tw))
            y0 = random.randint(0, max(1, h - th))
            tile = (x0, y0, x0+tw, y0+th)
            if all(iou(tile, gt) < 0.05 for gt in gts):
                crop = im[y0:y0+th, x0:x0+tw]
                out_img = out_dir / f"{img_path.stem}_bg{k}.jpg"
                out_img.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out_img), crop)
                # empty label file
                (out_lbl / f"{img_path.stem}_bg{k}.txt").write_text("")
                break
