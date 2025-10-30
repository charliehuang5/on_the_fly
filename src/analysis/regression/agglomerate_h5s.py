import h5py
import argparse
import glob
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    "--h5Dir", type=str, required=True, help="directory containing h5 datasets"
)
args = parser.parse_args()

h5Paths = glob.glob(args.h5Dir + "/**/activation_map*.h5", recursive=True)
if len(h5Paths) == 0:
    raise FileNotFoundError(f"no activation map datasets found within {args.h5Dir}")
bigH5 = h5py.File(args.h5Dir + "/all_activation_maps.h5", "w")

# iterate over h5 datasets
for path in h5Paths:
    fileName = os.path.basename(path)
    expName = fileName.split("__")[1].split(".")[0]  # name of this experiment
    smallH5 = h5py.File(path, "r")
    featNames = smallH5.keys()

    # update bigH5 data
    for featName in featNames:
        if featName not in bigH5:
            bigH5.create_group(featName)
        bigH5[featName].create_dataset(expName, data=smallH5[featName])

    # delete the original h5 file
    os.remove(path)

bigH5.close()
