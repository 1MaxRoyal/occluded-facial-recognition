import torch

#get cpu or gpu devce for training
device = "cuda" if torch.cuda.is_available() else "cpu"

#Load Model
model=torch.load("models/model.pth").to(device)
model.load_state_dict(torch.load('models/state.pth',map_location=device))
