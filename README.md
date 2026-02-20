This project is a simple Python script that performs basic image processing operations (color detection, contour detection, and image cropping/slicing) using OpenCV.

## Features

- **Image Loading**: Reads the image from the specified path.
- **Color Space Conversion**: Converts from BGR color space to HSV color space for more robust color-based segmentation.
- **Color Detection**: Creates a binary mask that isolates pixels within a specific color range (e.g., red).
- **Contour Detection**: Detects the outlines of objects on the mask and finds the object that occupies the largest area.
- **Image Slicing**: Crops the top-left corner of the original image (by 25% of the width and height).
- **Visualization**: Visualizes the detected and sliced ​​areas by drawing rectangles and text on the original image. - **Save**: Saves the cropped region (ROI) as a new image file (`output_crop.jpg`).

## Requirements

To run this project, the following libraries must be installed:

- Python 3.x
- OpenCV (`cv2`)
- NumPy

You can install the necessary packages using pip:

```bash
pip install opencv-python numpy or using requirements.txt file  
```

## Usage

To run the project, you can run the following command in the terminal or command prompt:

```bash
python main.py
```

The code will default to processing a file named `Red_Apple.jpg`. If you want to use a different file, you can update the file path in `main.py`.

## Output

When the script runs successfully, it will open three different windows (Binary Mask, Detection Result, Cropped ROI) to show the steps of the process and save a cropped image file named `output_crop.jpg` to the working directory. You can press any key to close the windows.