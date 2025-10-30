import sys 
sys.path.insert(0, "/tigress/MMURTHY/Max/courtship_dynamics")
from src.utils.movies import *
import pickle
import argparse
import glob
import os 
import ants

parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True, help="experiment directory")
parser.add_argument("--fdaPath", type=str, default="/Users/mjaragon/Desktop/registration_templates/templates/FDA.nii", help="path to FDA directory")
args = parser.parse_args()

# ----------------------------
# % load the regression data
# ----------------------------

regressionDataPath = glob.glob(args.dir + "/**/regression_results.pkl", recursive=True)
if not regressionDataPath:
    raise FileNotFoundError(f"no regression data found within {args.dir}")
else:
    regressionDataPath = regressionDataPath[0]

with open(regressionDataPath, "rb") as inFile:
    regressionData = pickle.load(inFile)

sigMod = regressionData["sigMod"]  # significantly modulated ROIs for each feature
featNames = list(sigMod.keys())  # regression feature names
idxToROI = regressionData["idxToROI"]  # map from python indices -> ROI labels

# ----------------------------------------------------------
# % load and downsample the FDA and warped supervoxel data
# ----------------------------------------------------------

supervoxelPath = glob.glob(args.dir + "/**/supervoxels_fda.nii", recursive=True)
if not supervoxelPath:
    raise FileNotFoundError(f"no supervoxel data found within {args.dir}")
else:
    supervoxelPath = supervoxelPath[0]

fdaPath = args.fdaPath
if not os.path.exists(fdaPath):
    raise FileNotFoundError(f"no FDA data found within {args.dir}")

print("Downsampling supervoxel and FDA data...")
downscale_factor = (0.5, 0.5, 0.5)  # dowsample by a factor of 2
supervoxels = ants.image_read(supervoxelPath).numpy().astype(np.uint16)
fda = ants.image_read(fdaPath).numpy()
supervoxelWorking = downsample(supervoxels, downscale_factor=downscale_factor, order=0)
fdaWorking = downsample(fda, downscale_factor=downscale_factor)
print("Done!")

# -------------------
# % make the movies
# -------------------

# create save dir
saveDir = os.path.dirname(regressionDataPath) + "/supervoxel_movies"
if not os.path.exists(saveDir):
    os.makedirs(saveDir)

print("Making supervoxel movies...")
for featName in featNames:
    try:
        modulated = [idxToROI[x] for x in sigMod[featName]]  # supervoxel labels 
    except:
        continue

    makeResponderMovie(
        modulated,
        supervoxelWorking,
        fdaWorking,
        output_file=saveDir + f"/{featName}__supervoxels.mp4",
        debug=False,
    )
print("Done!")
