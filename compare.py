import pickle as pkl
from torchvision.transforms import Compose, ToTensor
import numpy as np

from load_model import model, device
device = 'cpu'
model.to(device)
model.eval()
data = data = "data/database/"
    
transformer = Compose([
    np.float32,
    ToTensor(),
    ])


def compare(img):
    load_embeddings = pkl.load( open( data + "embeddings.pkl", "rb" ) )

    embeddings = []
    paths = []

    for le in load_embeddings:
        embeddings.append(le[0])
        paths.append(le[1])

    img = transformer(img)
    img_embed = model(img.unsqueeze(0))
    dists = [(img_embed - e).norm().item() for e in embeddings]
    #if min(dists) < .4:
    min_dist = dists.index(min(dists))
    pred = paths[min_dist]
    #else:
     #   pred = "None"

    return pred