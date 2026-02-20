"""Module for image processing, color detection, and image slicing.

This module uses OpenCV to detect red-colored objects in an image and
slices the upper-left corner for Region of Interest (ROI) analysis.
"""
import cv2
import numpy as np

def process_and_crop_object(image_path):
    """Processes an image to detect red objects and crops the upper-left corner.

    The function performs the following steps:
    1. Loads the image from the given path.
    2. Converts the image to HSV color space.
    3. Masks red colors and finds contours.
    4. Slices the top-left 25% of the image as a Region of Interest (ROI).
    5. Visualizes the ROI on the original image and saves the crop.

    Args:
        image_path (str): Absolute or relative path to the input image file.

    Returns:
        None: Displays results using OpenCV pindows and saves 'output_crop.jpg'.
    """
    # 1. Load the Image
    # 'image' is a NumPy array with shape (height, width, channels)
    src_image = cv2.imread(image_path)
    
    if src_image is None:
        print(f"Error: Could not find image at {image_path}")
        return

    # 2. Convert Color Space (BGR to HSV)
    # HSV is more robust for color-based segmentation
    hsv_image = cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV)

    # 3. Define Color Range for Detection (Red color wraps around 0 and 180)
    # Lower range: 0-10
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    # Upper range: 170-180
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Create two binary masks
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # 4. Use bitwise_or to combine both red masks
    mask = cv2.bitwise_or(mask1, mask2)

    # 5. Find Contours (Shapes) in the combined mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour based on area
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get Bounding Box coordinates: x, y (top-left), w (width), h (height)
        x, y, w, h = cv2.boundingRect(largest_contour)

        # 5. IMAGE SLICING (The core part of your question)
        # Slicing the upper corner (top-left) of the original image
        height, width = src_image.shape[:2]
        crop_h = int(height * 0.25)
        crop_w = int(width * 0.25)
        cropped_object = src_image[0:crop_h, 0:crop_w]

        # 6. Visualization (Drawing results on the original image)
        # cv2.rectangle(image, start_point(x,y), end_point(x,y), color(B,G,R), thickness)
        cv2.rectangle(src_image, (0, 0), (crop_w, crop_h), (0, 255, 0), 3)
        cv2.putText(src_image, "Upper Corner", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 7. Show and Save Results
        cv2.imshow('Binary Mask', mask)
        cv2.imshow('Detection Result', src_image)
        cv2.imshow('Cropped ROI', cropped_object)
        
        # Save the slice to a file
        cv2.imwrite('images/output_crop.jpg', cropped_object)
        print("Success: ROI (Region of Interest) has been saved.")

    else:
        print("No object detected within the specified color range.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# To run the simulation:
if __name__ == "__main__":
 process_and_crop_object('images/Red_Apple.jpg')