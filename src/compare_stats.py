import cv2
import numpy as np
import glob

# Load a few training images
train_files = glob.glob('data/images/train/*.jpg')[:50]
if not train_files:
    print("No train files found in data/images/train")
else:
    train_imgs = [cv2.imread(f, 0) for f in train_files]
    train_mean = np.mean([np.mean(x) for x in train_imgs])
    train_std = np.mean([np.std(x) for x in train_imgs])
    print(f'Train (NEU-DET) - Mean Intensity: {train_mean:.2f}, Std Dev: {train_std:.2f}')

# Load a test image 
test_files = glob.glob('test/*/*.jpg')[:10]
if not test_files:
    print("No test files found")
else:
    test_imgs = [cv2.imread(f, 0) for f in test_files if f is not None]
    # Approximate tiling stats (taking center crop stats)
    # Test images are 2048x1000. Center is around 500, 1024
    test_crops = [x[400:600, 924:1124] for x in test_imgs] # 200x200 crop
    test_mean = np.mean([np.mean(x) for x in test_crops])
    test_std = np.mean([np.std(x) for x in test_crops])
    print(f'Test (Current)  - Mean Intensity: {test_mean:.2f}, Std Dev: {test_std:.2f}')
