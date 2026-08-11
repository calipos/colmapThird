import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import bisenet


parser = bisenet.BiSeNet("models/faceparsingrResnet34.onnx")
test_images_dir = Path('data/a')
test_images = sorted(test_images_dir.glob('*.jpg'))

# Store original and processed images
original_images = []
parsed_images = []

for image_path in test_images:
    print(f"Processing: {image_path.name}")

    # Load image (already a face crop)
    image = cv2.imread(str(image_path))

    # Parse the face directly
    mask = parser.parse(image)
    unique_classes = len(set(mask.flatten()))
    print(f'  Parsed with {unique_classes} unique classes')

    # Visualize the parsing result
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    vis_result = bisenet.vis_parsing_maps(image_rgb, mask, save_image=False)

    cv2.imwrite(str(image_path)+'.bmp', vis_result)
    # original_images.append(image_rgb)
    # parsed_images.append(vis_result)

print(f"\nProcessed {len(test_images)} images")
