import os
import xml.etree.ElementTree as ET


# Paths
train_img_dir = "data/images/train/"
val_img_dir   = "data/images/val/"
train_label_dir = "data/labels/train/"
val_label_dir   = "data/labels/val/"
train_label_dir_o = "data/NEU-DET/train/annotations/"
val_label_dir_o   = "data/NEU-DET/validation/annotations/"

os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)

# Map class names to ids
class_map = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5
}

def process_split(img_dir, anno_dir, out_label_dir):
    jpgs = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]
    for filename in jpgs:
        xml_path = os.path.join(anno_dir, filename.replace(".jpg", ".xml"))
        out_path = os.path.join(out_label_dir, filename.replace(".jpg", ".txt"))

        if not os.path.exists(xml_path):
            print(f"Warning: missing XML for {filename}")
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            with open(out_path, "w") as f:
                for obj in root.findall("object"):
                    label = obj.findtext("name", "").strip()
                    if label not in class_map:
                        print(f"Warning: unknown class '{label}' in {xml_path}")
                        continue
                    class_id = class_map[label]

                    b = obj.find("bndbox")
                    xmin = int(b.findtext("xmin"))
                    ymin = int(b.findtext("ymin"))
                    xmax = int(b.findtext("xmax"))
                    ymax = int(b.findtext("ymax"))
                    # convert VOC -> YOLO
                    img_w = int(root.find("size/width").text)
                    img_h = int(root.find("size/height").text)
                    
                    x_center = ((xmin + xmax) / 2) / img_w
                    y_center = ((ymin + ymax) / 2) / img_h
                    box_w = (xmax - xmin) / img_w
                    box_h = (ymax - ymin) / img_h
                    
                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n")


        except ET.ParseError as e:
            print(f"Error parsing {xml_path}: {e}")

# Run for train and val
process_split(train_img_dir, train_label_dir_o, train_label_dir)
process_split(val_img_dir,   val_label_dir_o,   val_label_dir)

print("All label files generated successfully.")
