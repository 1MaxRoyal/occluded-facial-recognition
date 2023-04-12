import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
import numpy as np

import logger
log=logger.log('test')

testData = "data/datasets/fei_cropped_split/test"

transformer = Compose([
    np.float32,
    ToTensor(),
    ])
testDataset = datasets.ImageFolder(testData,transform=transformer)
testDataset.idx_to_class = {i:c for c, i in testDataset.class_to_idx.items()}

from load_model import model
model.num_classes =len(testDataset.class_to_idx)
model.to("cpu")
model.eval()
model.classify = True

correct = 0
total = 0

for img in range(len(testDataset)):
    x, y = testDataset[img][0], testDataset[img][1]
    
    names = []
    names.append(testDataset.idx_to_class)
    
    with torch.no_grad():
        pred = model(x.unsqueeze(0))
        _, output = torch.max(pred, 1)
        predicted, actual = names[0][output.item()], names[0][y]
        total += 1
        if predicted == actual:
            correct += 1
        print(f'P "{predicted}", A "{actual}"')
        
        
print(f'Predicted "{correct}" correct out of: "{total}"')
percent = (correct/total)*100
print(f'{percent}% correct')

#return output to console and close file
logger.end(log)