import pickle as pkl
from torchvision.transforms import Compose, ToTensor
import numpy as np

from load_model import model


def compare(img):
    device = 'cpu'
    model.to(device)
    model.eval()
    data = "data/database/"
        
    transformer = Compose([
        np.float32,
        ToTensor(),
        ])
    load_embeddings = pkl.load( open( data + "embeddings.pkl", "rb" ) )

    embeddings = []
    paths = []

    for le in load_embeddings:
        embeddings.append(le[0])
        paths.append(le[1])
    img = transformer(img).to(device)
    img_embed = model(img.unsqueeze(0))
    dists = [(img_embed - e).norm().item() for e in embeddings]
    print(min(dists))
    if min(dists) < 1:
        min_dist = dists.index(min(dists))
        pred = paths[min_dist]
    else:
        pred = None

    return pred