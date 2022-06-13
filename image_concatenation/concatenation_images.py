"""
How To Use?
1. Create "convert_test_image" directory and "test_data" directory.
2. Put images into "convert_test_image" directory.
(only 2 images: front image, back image)
3. Find "pill_number" in the CSV and replace the variable's value.
4. Code "python concatenation_images.py" in terminal.
"""

import cv2
import numpy as np
import os

image_root = "/opt/ml/final-project-level3-cv-16/convert_test_image"
save_root = "/opt/ml/final-project-level3-cv-16/test_data"
images = os.listdir(image_root)
pill_number = 199400579

image1 = cv2.imread(os.path.join(image_root, images[0]), 1)
h, w, _ = image1.shape
if w > h:
    image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(os.path.join(save_root, f"{pill_number}_{1}.jpg"), image1)

image2 = cv2.imread(os.path.join(image_root, images[1]), 1)
h, w, _ = image2.shape
if w > h:
    image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite(os.path.join(save_root, f"{pill_number}_{2}.jpg"), image2)

concat_image = np.concatenate((image1, image2), axis=1)
cv2.imwrite(os.path.join(save_root, f"{pill_number}_{3}.jpg"), concat_image)

os.remove(os.path.join(image_root, images[0]))
os.remove(os.path.join(image_root, images[1]))
