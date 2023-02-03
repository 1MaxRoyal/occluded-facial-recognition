import torch
from facenet_pytorch import InceptionResnetV1

#Download Model from pytorch-facenet that has been pretrained on vggface2 dataset
#https://github.com/timesler/facenet-pytorch
model = InceptionResnetV1(pretrained='vggface2')
torch.save(model,"../models/model1.pth")
torch.save(model.state_dict(),"../models/state1.pth")