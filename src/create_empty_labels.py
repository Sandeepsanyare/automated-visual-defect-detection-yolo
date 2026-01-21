# create_empty_labels.py
from pathlib import Path

img_dir = Path("datasets/yourset/images/train")  # repeat for val if needed
lbl_dir = Path("datasets/yourset/labels/train")
lbl_dir.mkdir(parents=True, exist_ok=True)

exts = {".jpg", ".jpeg", ".png", ".bmp"}
images = [p for p in img_dir.rglob("*") if p.suffix.lower() in exts]

made = 0
for img in images:
    rel = img.relative_to(img_dir).with_suffix(".txt")
    lbl = lbl_dir / rel
    if not lbl.exists():
        lbl.parent.mkdir(parents=True, exist_ok=True)
        lbl.write_text("")  # empty file means no objects
        made += 1
print(f"Created {made} empty labels")
