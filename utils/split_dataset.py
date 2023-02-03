#splits the dataset into test, train and val
import splitfolders
dataDir = "../data/datasets/fei_cropped"
splitfolders.ratio(dataDir, output=dataDir + "_split",
    seed=1006, ratio=(.8, .1, .1), group_prefix=None, move=False)
