import cv2 as cv
import numpy as np
from torchvision import datasets, transforms
import os

def skin_detector(img):
    # Convert the image to the YCbCr color space
    ycbcr = cv.cvtColor(img, cv.COLOR_BGR2YCrCb)
    
    # Define the range of skin color in the YCbCr color space and create mask
    lower = np.array([0, 135, 85], dtype=np.uint8)
    upper = np.array([255, 180, 135], dtype=np.uint8)
    skin_mask = cv.inRange(ycbcr, lower, upper)
    
    # Apply the skin mask to the image and convert to grayscale
    skin = cv.bitwise_and(img, img, mask=skin_mask)
    gray = cv.cvtColor(skin, cv.COLOR_BGR2GRAY)

    # Apply morphological operations to remove non-skin pixels
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    binary = cv.erode(binary, kernel, iterations=1)
    binary = cv.dilate(binary, kernel, iterations=1)
    binary = cv.bitwise_not(binary)
    
    # Apply the mask to the original image
    skin = cv.bitwise_and(img, img, mask=binary)   
        
    return skin


def skin_cropper(img):
    # Apply skin color segmentation to the image
    skin = skin_detector(img)

    gray = cv.cvtColor(skin, cv.COLOR_BGR2GRAY)
    
    # Find contours in the grayscale image
    contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    # Find the contour with the largest area
    max_contour = max(contours, key=cv.contourArea)
    
    # Get the bounding rectangle of the contour
    x, y, w, h = cv.boundingRect(max_contour)
    
    # Crop the image to the bounding rectangle
    cropped = img[y:y+h, x:x+w]
    
    # resize to 160x160 because that is what the model is trained on
    cropped = cv.resize(cropped,(160,160))
    
    return cropped

# images that you want to be cropped to face
data = "../data/datasets/fei"

# dataset images
dataset= datasets.ImageFolder(data, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data, data))
        for p, _ in dataset.samples
]

# create directory for cropped images
if not os.path.exists(data + "_cropped"):
    os.mkdir(data + "_cropped")

# crop and save every image
# ignoring images with the suffix 14.jpg as they are image with reduced lighting and we are not interested in that
for i, (x, y) in enumerate(dataset):
    if not y.endswith("14.jpg"):
        img = cv.imread(y)
        img = skin_cropper(img)
        q = y.replace(data,data + "_cropped")
        if not os.path.exists(os.path.dirname(q)):
            os.mkdir(os.path.dirname(q))
        cv.imwrite(q,img)
        print(y)
        print(q)

print("Done!")