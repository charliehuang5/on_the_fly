import h5py 
import argparse
import os 

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True, help="data directory")
args = parser.parse_args()

# create h5 file 
datasetPath = args.dir + "/activation_map_dataset.h5"
if os.path.exists(datasetPath): 
    os.remove(datasetPath)
h5Dataset = h5py.File(datasetPath, "w")
h5Dataset.close()
