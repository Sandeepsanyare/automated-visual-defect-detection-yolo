import cv2
import numpy as np
import os

def check_stats(path, is_tile_needed=False):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    img = cv2.imread(path, 0)
    if img is None:
        print(f"Failed to read: {path}")
        return

    if is_tile_needed:
        # Simulate a 200x200 tile from the center
        h, w = img.shape
        cy, cx = h//2, w//2
        crop = img[cy-100:cy+100, cx-100:cx+100]
        mean_val = np.mean(crop)
        std_val = np.std(crop)
    else:
        mean_val = np.mean(img)
        std_val = np.std(img)
        
    print(f"File: {os.path.basename(path)}")
    print(f"  Mean Intensity: {mean_val:.2f}")
    print(f"  Contrast (Std): {std_val:.2f}")

if __name__ == "__main__":
    print("--- Training Image ---")
    check_stats("data/images/train/crazing_151_bg2.jpg")
    
    print("\n--- Test Image (Center Tile) ---")
    check_stats("test/1/img_02_425501900_00017.jpg", is_tile_needed=True)
