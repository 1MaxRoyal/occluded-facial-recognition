import torch
torch.cuda.empty_cache()
import gc
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import datasets, transforms
import numpy as np
import os

from trainer import train
import logger

log=logger.log('train')

#get model and device to train on, gpu or cpu
from load_model import model, device
print(f"\nUsing {device} device")

#setup data for datasets
trainData = "data/datasets/fei_cropped_split/train"
valData = "data/datasets/fei_cropped_split/val"
batchSize = 16
epochs  = 8
workers = 0 if os.name == 'nt' else 8

#transforms to apply to datasets
transformer = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
])

#compile data into datasets and apply transforms
trainDataset = datasets.ImageFolder(trainData, transform=transformer)
valDataset = datasets.ImageFolder(valData, transform=transformer)     

#place the datasets into dataloaders
trainLoader = DataLoader(
    trainDataset,
    num_workers=workers,
    batch_size=batchSize,
)

valLoader = DataLoader(
    valDataset,
    num_workers=workers,
    batch_size=batchSize,
    )

#setup model for dataset
model.classify = True
#both datasets have the same number of classes so it doesnt matter if you put val or train dataset here
model.num_classes =len(trainDataset.class_to_idx)

#define optimizer and scheduler
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = MultiStepLR(optimizer, [5, 10])

#define loss function
lossFunc = torch.nn.CrossEntropyLoss()

#train model
print('\n\nInitial')
print('-' * 10)

model.eval()
train(model,lossFunc,valLoader, mode = 'val', device=device)
for epoch in range(epochs):
    print('\nEpoch {}/{}'.format(epoch + 1, epochs))
    print('*' * 5)

    model.train()
    train(model,lossFunc,trainLoader,'train',optimizer,scheduler,device)
    
    model.eval()
    train(model,lossFunc,valLoader, mode = 'val', device=device)

#save model
torch.save(model.state_dict(),"models/state.pth")
torch.cuda.empty_cache()
gc.collect()

#return output to console and close file
logger.end(log)