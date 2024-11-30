'''
# METHOD 1

import cv2
import os
import numpy as np

def detect_coral_babies(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to smooth the image and reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding to binarize the image
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Find contours in the binary image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter the contours to find the ones with an approximate circular shape
            coral_babies = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if 0.5 <= circularity <= 0.9:
                        coral_babies.append(cnt)

            # Draw the detected coral babies on the original image
            output_img = img.copy()
            for cnt in coral_babies:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output_img, center, radius, (0, 255, 0), 2)

            # Save the output image with detected coral babies
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, output_img)
            print(f"Processed and saved image with {len(coral_babies)} coral babies detected: {output_path}")

def main():
    input_dir = "original_images/"
    output_dir = "4.Hough_Transform/"
    detect_coral_babies(input_dir, output_dir)

if __name__ == "__main__":
    main()
'''
###########################################################################################################################

# METHOD 2

import cv2
import os
import numpy as np

def detect_coral_babies(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".jpg"):
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Failed to load image: {filename}")
                continue

            # Convert the image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to smooth the image and reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Apply adaptive thresholding to binarize the image
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

            # Find contours in the binary image
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Filter the contours to find the ones with an approximate circular shape
            coral_babies = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                if 20 < perimeter < 5000 and 50 < area < 20000:
                    circularity = 4 * np.pi * area / (perimeter ** 2)
                    if 0.5 <= circularity <= 0.9:
                        coral_babies.append(cnt)

            # Draw the detected coral babies on the original image
            output_img = img.copy()
            for cnt in coral_babies:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius)
                cv2.circle(output_img, center, radius, (0, 255, 0), 2)

            # Save the output image with detected coral babies
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, output_img)
            print(f"Processed and saved image with {len(coral_babies)} coral babies detected: {output_path}")

def main():
    input_dir = "original_images/"
    output_dir = "4.Hough_Transform/"
    detect_coral_babies(input_dir, output_dir)

if __name__ == "__main__":
    main()

###########################################################################################################################
'''
# METHOD 3

import cv2
import numpy as np

# Load the image
image = cv2.imread('original_images/32_T2_6_timepoint0.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply GaussianBlur to reduce noise and improve circle detection
blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# Use HoughCircles to detect circles in the image
circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                           param1=50, param2=30, minRadius=50, maxRadius=100)

# If some circles are detected, let's draw them on the original image
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    # Print the number of detected coral babies
    print(f"Number of detected coral babies: {len(circles)}")

# Save the result image
cv2.imwrite('4.Hough_Transform/new1.jpg', image)

print("The image with detected coral babies has been saved as 'coral_babies_detected.jpg'.")

'''
###########################################################################################################################
'''
# METHOD 4

import cv2
import numpy as np

def detect_circular_blobs(image_path, debug=False):
    """
    Detect and count circular blobs in an image that resemble small floral clusters or bacteria.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image file
    debug : bool, optional
        If True, displays intermediate processing steps (default is False)
    
    Returns:
    --------
    tuple: (processed_image, blob_count, blob_details)
        - processed_image: Image with detected blobs highlighted
        - blob_count: Number of detected blobs
        - blob_details: List of blob information (center, radius, etc.)
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Create a copy for processing and drawing
    processed = image.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Adaptive thresholding to handle varying background
    thresh = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11,  # block size
        2    # constant subtracted from mean
    )
    
    if debug:
        cv2.imshow('Thresholded Image', thresh)
        cv2.waitKey(0)
    
    # Find contours
    contours, _ = cv2.findContours(
        thresh, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    # Blob detection parameters
    blobs = []
    for contour in contours:
        # Calculate contour area and perimeter
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Skip very small or very large contours
        if area < 10 or area > image.shape[0] * image.shape[1] * 0.1:
            continue
        
        # Calculate circularity
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
        else:
            continue
        
        # Check if the blob is sufficiently circular
        if circularity > 0.7:  # Adjust this threshold as needed
            # Get the minimal enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            
            # Additional color consistency check
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            # Calculate mean and standard deviation of pixel intensities inside and around the blob
            mean_inside = cv2.mean(gray, mask=mask)[0]
            mean_outside = cv2.mean(gray, mask=cv2.bitwise_not(mask))[0]
            
            # Check color variation
            if abs(mean_inside - mean_outside) < 30:  # Adjust threshold as needed
                blobs.append({
                    'center': center,
                    'radius': radius,
                    'area': area,
                    'circularity': circularity
                })
                
                # Draw the detected blob
                cv2.circle(processed, center, radius, (0, 255, 0), 2)
                cv2.circle(processed, center, 3, (255, 0, 0), -1)
    
    if debug:
        cv2.imshow('Detected Blobs', processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return processed, len(blobs), blobs

# Example usage
def main():
    # Replace with your image path
    image_path = 'original_images/32_T2_6_timepoint0.jpg'
    
    try:
        # Detect blobs with debug mode on
        result_image, blob_count, blob_details = detect_circular_blobs(image_path, debug=True)
        
        print(f"Number of blobs detected: {blob_count}")
        
        # Print details of each blob
        for i, blob in enumerate(blob_details, 1):
            print(f"Blob {i}:")
            print(f"  Center: {blob['center']}")
            print(f"  Radius: {blob['radius']}")
            print(f"  Area: {blob['area']}")
            print(f"  Circularity: {blob['circularity']:.2f}")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
'''
