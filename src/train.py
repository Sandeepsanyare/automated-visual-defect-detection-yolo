from ultralytics import YOLO

# Phase 1
model = YOLO("../models/yolov8s.pt")
model.train(
    data="data.yaml",
    epochs=350,
    patience=40,
    imgsz=320,
    batch=256,
    cos_lr=True,
    # geometric
    degrees=10.0,      # small rotations
    translate=0.10,    # up to 10 percent shift
    scale=0.50,        # up to 50 percent zoom in or out
    shear=2.0,
    fliplr=0.5,        # horizontal flip
    # color
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    # mix augment
    mosaic=1.0, mixup=0.10, copy_paste=0.10,
    close_mosaic=15,   # disable mosaics near the end
    project="results",
    name="yolov8s_defect_detector_aug",
    workers=2,
)

# Phase 2 fine tune for localization
best = "results/yolov8s_defect_detector_aug/weights/best.pt"
YOLO(best).train(
    data="data.yaml",
    epochs=15,
    patience=8,
    imgsz=448,         # a bit larger
    batch=256,
    lr0=0.001, lrf=0.0005, cos_lr=True,
    # very light aug so boxes settle
    mosaic=0.0, mixup=0.0, copy_paste=0.0,
    degrees=0.0, translate=0.02, scale=0.20, shear=0.0,
    fliplr=0.5,
    close_mosaic=5,
    project="results",
    name="yolov8s_defect_detector_aug_finetune",
    workers=2,
)
