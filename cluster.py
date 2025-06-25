import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load the image
image = cv2.imread("crack_image.jpg")  # Replace with your image path
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for visualization

# Step 2: Preprocess the image (convert to grayscale for simplicity)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

kernel_size = 5  # The kernel size must be an odd number
gray_image = cv2.medianBlur(gray_image, kernel_size)

clahe = cv2.createCLAHE(clipLimit=100.0, tileGridSize=(8, 8))  # Adjust clipLimit for more/less contrast
enhanced_image = clahe.apply(gray_image)
# Step 3: Flatten the image (reshape into a 2D array of pixels)
pixels = enhanced_image.reshape((-1, 1))  # Reshape to a 2D array (each pixel is one sample)

# Step 4: Perform K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters: background and crack
kmeans.fit(pixels)

# Step 5: Assign cluster labels to each pixel
segmented_image = kmeans.labels_.reshape(gray_image.shape)  # Reshape to the original image size

# Step 6: Visualize the clustered image
plt.figure(figsize=(10, 5))

# Show the original image
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")

# Show the segmented (clustered) image
plt.subplot(1, 2, 2)
plt.imshow(segmented_image, cmap='gray')
plt.title("Clustered Image (Crack Segmentation)")
plt.axis("off")

plt.show()

# Optionally: Save the segmented result
cv2.imwrite("segmented_crack.jpg", segmented_image * 255)  # Save the segmented image
