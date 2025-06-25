import cv2
import numpy as np
import matplotlib.pyplot as plt
import localthickness as lt
import random

def localThickness(image):
    return lt.local_thickness(image, scale=1),lt.local_thickness(~image, scale=1)
def avg(thickness):
    sum = 0
    for i in range(len(thickness)):
        sum += thickness[i]
    return sum/len(thickness)
def computeThickness(binary_image, sample_rows=10000):
    rows, cols = binary_image.shape
    sampled_rows = random.sample(range(rows), min(sample_rows, rows))
    
    thickness = []
    for row in sampled_rows:
        thick = 0
        for i in range(len(binary_image[row])):
            if(binary_image[row][i]==0):
                thick+=1
        thickness.append(thick)
    
    return avg(thickness)
        
def find_crack_widths(binary_image_path, scaleFactor=1):
    
    binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

     
    crack_measurements = []


    top_left = None
    height, width = binary_image.shape
    max_square_size = 0
    cumsum = np.cumsum(np.cumsum(binary_image == 0, axis=0), axis=1)
    i=0
    for y in range(height):
        for x in range(width):
            # Maximum possible square size from this point
            max_possible_size = min(height - y, width - x)
            for size in range(max_possible_size, 0, -1):
                # Bottom-right corner of the square
                br_y, br_x = y + size - 1, x + size - 1
                print("check :", i)
                i+=1
                # Calculate the number of zeros in the square using cumulative sum
                total_zeros = cumsum[br_y, br_x]
                if y > 0:
                    total_zeros -= cumsum[y - 1, br_x]
                if x > 0:
                    total_zeros -= cumsum[br_y, x - 1]
                if y > 0 and x > 0:
                    total_zeros += cumsum[y - 1, x - 1]
                
                # Total elements in the square
                total_elements = size * size
                if total_zeros / total_elements >= 0.9:
                    if size > max_square_size:
                        max_square_size = size
                        top_left = (x, y)
                    break

    if max_square_size > 0 and top_left:
        crack_measurements.append({
            "max_square_width": max_square_size * scaleFactor,
            "bounding_box": {
                "top_left": top_left,
                "bottom_right": (top_left[0] + max_square_size, top_left[1] + max_square_size)
            }
        })

    return crack_measurements



def LoadImage(path = "crack_image1.jpg", actualHeight = 1):
    
    image = cv2.imread(path)  
    height, width, channels = image.shape
    scaleFactor = actualHeight/height
    
    print(f"Image Width: {width} pixels")
    print(f"Image Height: {height} pixels")
    print(f"Number of Channels: {channels}")


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold_value = 80  
    _, binary_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    output_path = "thresholded.png"
    cv2.imwrite(output_path, binary_image)
  
    return image_rgb, binary_image, scaleFactor

def plotImage(image_rgb, binary_image):
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis("off")


    plt.subplot(1, 2, 2)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image (Global Thresholding)")
    plt.axis("off")
    plt.show()

    cv2.imwrite("binary_image.jpg", binary_image) 

def draw_bounding_boxes(image_path, bounding_boxes):
   
    image = cv2.imread(image_path)
    for box in bounding_boxes:
        top_left = box["bounding_box"]["top_left"]
        bottom_right = box["bounding_box"]["bottom_right"]
        cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

 
    cv2.imshow("Image with Bounding Boxes", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("output_with_boxes.jpg", image)


def main():
    
    actualHeight = float(input("Enter the measured height of the image bounds in centimetres: "))
    image_rgb, binary_image, scaleFactor = LoadImage("crack_image1.jpg", actualHeight)
    
    plotImage(image_rgb, binary_image)
    
    binary_image_path = "thresholded.png"  
    #crack_measurements = find_crack_widths(binary_image_path, scaleFactor)
    #print(crack_measurements)
    
    #draw_bounding_boxes(binary_image_path, crack_measurements)
    #thickness, separation = localThickness(binary_image)
    #fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    #ax[0].imshow(binary_image[10])
    #ax[1].imshow(thickness[10], cmap=lt.black_plasma())
    #ax[2].imshow(separation[10], cmap=lt.white_viridis())
    
    thickness = computeThickness(binary_image)
    print("Width :", round(thickness*scaleFactor,2), "cm")


if __name__ =="__main__":
    main()
