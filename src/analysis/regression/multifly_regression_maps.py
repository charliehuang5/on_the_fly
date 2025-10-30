import sys

sys.path.insert(0, "/tigress/MMURTHY/Max/courtship_dynamics")
from src.utils.movies import *
import pickle
import argparse
import glob
import os
import ants
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import h5py

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dir", type=str, required=True, help="directory containing all experiments"
)
parser.add_argument(
    "--fdaPath",
    type=str,
    default="/Users/mjaragon/Desktop/registration_templates/templates/FDA.nii",
    help="path to FDA directory",
)
args = parser.parse_args()

# --------------
# % parameters
# --------------

# downscale_factor = (0.5, 0.5, 0.5)  # dowsample by a factor of 2
downscale_factor = (0.33, 0.33, 0.33)  # dowsample by a factor of 2

# ---------------------------
# % get experiment subdirs
# ---------------------------

regressionDataPaths = glob.glob(args.dir + "/**/regression_results.pkl", recursive=True)
expDirs = [str(Path(path).parents[3]) for path in regressionDataPaths]

# ---------------------
# % load the fda data
# ---------------------

print("loading and downsampling FDA data...")
fdaPath = args.fdaPath
if not os.path.exists(fdaPath):
    raise FileNotFoundError(f"no FDA data found at {args.fdaPath}")

fda = ants.image_read(fdaPath).numpy()
fdaWorking = downsample(fda, downscale_factor=downscale_factor)
# fdaWorking = fda
print("Done!")

# ---------------------------
# % instantiate h5 dataset
# ---------------------------
# h5Path = args.dir + "/multifly_labels.h5"
# if os.path.exists(h5Path):
#     os.remove(h5Path)
# h5Dset = h5py.File(h5Path)

# -------------------------------------------------
# % load the regression data for each experiment
# -------------------------------------------------

supervoxelMaps = defaultdict(dict)
print("collecting activation maps...")

for i, expDir in enumerate(tqdm(expDirs)):
    expName = expDir.split("/")[-1]
    regressionDataPath = glob.glob(
        expDir + "/**/regression_results.pkl", recursive=True
    )
    if not regressionDataPath:
        continue
    else:
        regressionDataPath = regressionDataPath[0]

    with open(regressionDataPath, "rb") as inFile:
        regressionData = pickle.load(inFile)

    sigMod = regressionData["sigMod"]  # significantly modulated ROIs for each feature
    featNames = list(sigMod.keys())  # regression feature names
    idxToROI = regressionData["idxToROI"]  # map from python indices -> ROI labels

    # if i==0:
    #     for featName in featNames:
    #         h5Dset.create_group(featName)

    # ----------------------------------------------------------
    # % load warped supervoxel data
    # ----------------------------------------------------------

    supervoxelPath = glob.glob(expDir + "/**/supervoxels_fda.nii", recursive=True)
    if not supervoxelPath:
        continue
    else:
        supervoxelPath = supervoxelPath[0]

    supervoxels = ants.image_read(supervoxelPath).numpy().astype(np.uint16)
    supervoxelWorking = downsample(
        supervoxels, downscale_factor=downscale_factor, order=0
    )

    # ------------------------------------------------------
    # % get supervoxel map for each regression feature
    # ------------------------------------------------------
    for featName in featNames:
        canvas = np.zeros_like(fdaWorking)
        try:
            modulated = [
                idxToROI[x] for x in sigMod[featName]
            ]  # supervoxel labels (not roi index)
        except:
            try:
                modulated = [idxToROI[int(sigMod[featName])]]  # single element
            except:
                continue

        # create activation map using regression coefficients
        featCoefs = regressionData["modelParams"][featName]
        try:
            activationCoefs = [
                featCoefs[idx] for idx in sigMod[featName]
            ]  # these are the coefs corresponding to the modulated ROIs
        except:
            try:
                activationCoefs = [featCoefs[sigMod[featName]]]
            except:
                continue

        activationMap = np.zeros_like(supervoxelWorking, dtype=np.float32)
        # modulatedVoxels = []

        for i, m in enumerate(tqdm(modulated)):
            whereSupervoxel = (
                supervoxelWorking == m
            )  # voxels within the modulated supervoxel
            # whereSupervoxel = np.where(supervoxelWorking==m)
            # modulatedVoxels.append(whereSupervoxel)
            # import pdb; pdb.set_trace() 
            activationMap[whereSupervoxel] = float(activationCoefs[i])  # coef corresponding to this supervoxel
        activationMap[activationMap==0] = np.nan
        supervoxelMaps[featName][expName] = activationMap
        # h5Dset[featName].create_dataset(expName, [modulatedVoxels, activationCoefs])
        # import pdb; pdb.set_trace() 

with open(args.dir + "/multifly_labels.pkl", "wb") as outFile:
    pickle.dump(supervoxelMaps, outFile)
# h5Dset.close()
print("Done!")
