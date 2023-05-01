# this script will create partially occluded images from the dataset, however on the prerequisite that the image has 
# already been cropped to the face and is 160x160 pixels
import cv2 as cv
from torchvision import datasets, transforms
import os

def occlude(c,img):
    occluded = {
        0: lambda img: cv.rectangle(img,(0,0),(80,80),(0,0,0), -1),
        1: lambda img: cv.rectangle(img,(80,0),(160,80),(0,0,0), -1),
        2: lambda img: cv.rectangle(img,(0,80),(80,160),(0,0,0), -1),
        3: lambda img: cv.rectangle(img,(80,80),(160,160),(0,0,0), -1),
        4: lambda img: cv.rectangle(img,(0,0),(160,80),(0,0,0), -1),
        5: lambda img: cv.rectangle(img,(0,80),(160,160),(0,0,0), -1),
        6: lambda img: cv.rectangle(img,(0,0),(80,160),(0,0,0), -1),
        7: lambda img: cv.rectangle(img,(80,0),(160,160),(0,0,0), -1),
        }[c](img)
    return occluded
        

# images that you want to be occluded
data = "../data/datasets/fei_cropped"

# dataset images
dataset= datasets.ImageFolder(data, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data, data))
        for p, _ in dataset.samples
]

# create directory for occluded images
if not os.path.exists(data + "_occluded_small"):
    os.mkdir(data + "_occluded_small")
    
# ignoring images with the suffix 14.jpg as they are image with reduced lighting and we are not interested in that
for i, (x, y) in enumerate(dataset):
    if not y.endswith("14.jpg"):
        for c in range(8):
            img = cv.imread(y)
            # code to occlude
            img = occlude(c,img)            
            q = y.replace(data,data + "_occluded_small")
            q = q.replace(".jpg", f"({str(c)}).jpg")
            if not os.path.exists(os.path.dirname(q)):
                os.mkdir(os.path.dirname(q))
            cv.imwrite(q,img)
            print(y)
            print(q)

print("Done!")