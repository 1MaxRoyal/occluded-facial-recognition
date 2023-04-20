import torch
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor
import numpy as np
import pickle as pkl
import cropper

from load_model import model, device
#device = 'cpu'
model.to(device)

class ImageFolderWithPaths(datasets.ImageFolder):

    def __getitem__(self, index):
  
        img, label = super(ImageFolderWithPaths, self).__getitem__(index)
        
        path = self.imgs[index][0]
        
        return (img, label ,path)

data = "data/database/"

transformer = Compose([
    np.float32,
    ToTensor(),
    ])

dataset = ImageFolderWithPaths(data, transform = transformer)
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}

names = []
images = []
paths = []

for img in range(len(dataset)):
    x, y, z = dataset[img][0], dataset[img][1], dataset[img][2]
    
    names.append(dataset.idx_to_class[y])
    images.append(x)
    paths.append(z)

images = torch.stack(images).to(device)
embeddings = model(images).detach().cpu()

save_embed = []
i=0
for e in embeddings:   
    temp = []
    temp.append(e)
    temp.append(paths[i])
    save_embed.append(temp)
    i += 1 
        
pkl.dump(save_embed,open(data + "embeddings.pkl", "wb"))
torch.cuda.empty_cache()
print("Embeddings Generated")