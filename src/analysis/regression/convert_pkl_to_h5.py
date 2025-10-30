import h5py
import pickle
from tqdm import tqdm

dataPath = "/Users/mjaragon/Desktop/multifly_labels.pkl"
h5Path = "/Users/mjaragon/Desktop/multifly_labels.h5"

with open(dataPath, 'rb') as inFile:
    data = pickle.load(inFile)

with h5py.File(h5Path, "w") as inFile:
    for featName in tqdm(data.keys()):
        expGroup = inFile.create_group(featName)
        for expName in data[featName].keys():
            expGroup.create_dataset(expName, data=data[featName][expName])
    
