import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the image
image = cv2.imread("crack_image.jpg")  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# Step 2: Preprocess the image (convert to grayscale for simplicity)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_size = 11  # The kernel size must be an odd number
gray_image = cv2.medianBlur(gray_image, kernel_size)

clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))  # Adjust clipLimit for more/less contrast
enhanced_image = clahe.apply(gray_image)
# Step 6: Visualize the clustered image
plt.figure(figsize=(10, 5))

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(enhanced_image)
plt.title("Original Image")
plt.axis("off")
plt.show()