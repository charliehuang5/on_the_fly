import pickle
from src.utils.movies import *
import numpy as np
from matplotlib import pyplot as plt 
import sys
from tqdm import tqdm
import os 
import ants
import cv2
import nibabel as nib
from matplotlib.colors import Normalize
from scipy.ndimage import label
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.stats import zscore
import h5py
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--h5Path',type=str,required=True,help='path to activation map dataset')
parser.add_argument('--fdaPath',type=str,required=True,help='path to FDA template')
args = parser.parse_args()

def getMaxProj(im):
    imWorking = im.copy()
    imWorking[np.isnan(imWorking)] = 0
    maxInds = np.nanargmax(abs(imWorking), axis=-1)
    maxProj = np.take_along_axis(imWorking, maxInds[..., None], axis=-1).squeeze(-1)
    maxProj[maxProj == 0] = np.nan
    return maxProj


def drawImageContour(
        image, doOverlay=False, binarize=True, contourType="boundary", thresh=117
):
    assert contourType in [
        "boundary",
        "internal",
    ], f"invalid contour type specified: {contourType}"
    imageWorking = image.copy(order="C")
    imageWorking = (
        255
        * (imageWorking - np.nanmin(imageWorking))
        / (np.max(imageWorking) - np.nanmin(imageWorking))
    ).astype(np.uint8)
    if contourType == "boundary":
        contourType = cv2.THRESH_BINARY
    elif contourType == "internal":
        contourType = cv2.RETR_EXTERNAL

    if binarize:
        imageWorking = cv2.equalizeHist(imageWorking)
        _, imageWorking = cv2.threshold(imageWorking, thresh, 255, cv2.THRESH_BINARY)
    if doOverlay:
        target = imageWorking.copy()
    else:
        target = np.zeros_like(image)
        contours, hierarchy = cv2.findContours(
            imageWorking, contourType, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(target, contours, -1, -1, 4)  # Draw all outer contours
    return target


def adaptive_alpha(intensity_values):
    """
    Create an adaptive alpha map based on intensity values.

    Parameters:
    intensity_values (numpy.ndarray): Input array of intensities.

    Returns:
    numpy.ndarray: An array of alpha values (transparency) based on the intensities.
                   Higher intensity values will have higher alpha (less transparent).
    """
    ivWorking = intensity_values.copy()
    ivWorking[np.isnan(ivWorking)] = 0
    abs_intensity = np.abs(ivWorking)
    norm = Normalize(vmin=0, vmax=np.nanmax(abs_intensity))  # Normalize to [0, 1]
    norm_values = norm(abs_intensity)  # Apply normalization

    # Invert the normalized values so lower intensity has more transparency (closer to 0)
    alpha = norm_values  # More intense = less transparent (closer to 1)

    return alpha


def getFeatGlob(supervoxelData, featName):
    # init globbed data
    featData = supervoxelData[featName]
    featGlob = []

    # init globbed data
    featGlob = []

    for expName in tqdm(featData.keys()):
        expData = featData[expName]
        expData[np.isnan(expData)] = 0  # convert nans to 0
        expData = np.where(abs(expData) > 0, 1, 0)  # binarize
        featGlob.append(expData.copy())

    # make feat glob image
    featGlob = np.nansum(featGlob, axis=0).astype(np.float32)
    return featGlob


def getSpatialCorrs(maps):
    map1, map2 = maps
    map1, map2 = map1.flatten(), map2.flatten()
    map1 = map1 - np.mean(map1)
    map2 = map2 - np.mean(map2)
    map1Std = np.std(map1)
    map2Std = np.std(map2)
    if map1Std==0 or map2Std==0:
        return 0 
    corr = np.dot(map1, map2) / (np.linalg.norm(map1) * np.linalg.norm(map2))
    return np.clip(corr, -1, 1)


def compute_fft_correlation(array1, array2):
    # Ensure arrays are the same size by padding to the larger shape
    shape = tuple(max(s1, s2) for s1, s2 in zip(array1.shape, array2.shape))
    padded1 = np.pad(array1, [(0, s - array1.shape[i]) for i, s in enumerate(shape)])
    padded2 = np.pad(array2, [(0, s - array2.shape[i]) for i, s in enumerate(shape)])

    # Compute FFTs
    fft1 = fftn(padded1)
    fft2 = fftn(padded2)

    # Cross-correlation in Fourier space
    cross_corr = fftshift(ifftn(fft1 * np.conj(fft2)))

    # Crop back to original dimensions
    start = tuple((p - o) // 2 for p, o in zip(cross_corr.shape, array1.shape))
    end = tuple(start[i] + array1.shape[i] for i in range(len(start)))
    cropped_corr = cross_corr[start[0] : end[0], start[1] : end[1], start[2] : end[2]]

    # Normalize the result
    cropped_corr = cropped_corr / (np.linalg.norm(array1) * np.linalg.norm(array2))

    return np.abs(cropped_corr)


def shufflePixels(image, mask):
    brainPixels = image[mask]
    np.random.shuffle(brainPixels)
    imageShuff = np.zeros_like(image)
    imageShuff[mask] = brainPixels
    return imageShuff

# dataPath = "/Users/mjaragon/Desktop/multifly_labels.pkl"
# with open(dataPath, "rb") as inFile:
#     supervoxelData = pickle.load(inFile)
dataPath = args.h5Path
supervoxelData = h5py.File(dataPath, "r")
fdaPath = args.fdaPath
fda = ants.image_read(fdaPath)
fda = fda.numpy()
fdaProj = np.max(fda, axis=-1)
fdaDS = downsample(fdaProj, downscale_factor=(0.33, 0.33))

# define cmap
cmap = plt.cm.get_cmap("coolwarm")
custom_colors = cmap(np.linspace(0, 1, 256))
custom_colors[0] = [0, 0, 0, 1]  # Set the color for the special value to black
myCmap = ListedColormap(custom_colors)
myCmap.set_bad(color="white")  # Set NaNs to desired color
myCmap.set_under("black")  # Values < vmin => black

# save dir
movieDir = os.path.dirname(args.h5Path) + "/activation_maps"
if not os.path.exists(movieDir):
    os.makedirs(movieDir)

# get fda contours
edge = drawImageContour(fdaDS, thresh=100, doOverlay=False)
internal = drawImageContour(
    fdaDS, doOverlay=False, binarize=True, contourType="internal", thresh=180
)

# bad experiments
# badExperiments = [
#     "20240606_101_func",
#     "20240606_202_func",
#     "20240606_102_func",
#     "20240618_201_func",
#     "20240618_101_func",
#     "20240618_102_func",
#     "20240612_201_func",
#     "20240623_201_func",
#     "20240617_401_func",
#     "20240617_201_func",
#     "20240609_101_func",
#     "20240610_201_func",
#     "20240610_101_func",
# ]
badExperiments = []

# set subplot params
nCols = 5

for featName in tqdm(supervoxelData.keys()):
    # init globbed data
    featData = supervoxelData[featName]
    featGlob = []

    # set figure
    nRows = len(featData.keys()) // nCols
    addRow = len(featData.keys()) % nCols != 0

    if addRow:
        nRows += 1

    fig, ax = plt.subplots(nRows, nCols)

    # init globbed data
    featGlob = []

    for i, expName in enumerate(featData.keys()):
        if "07" in expName or "20240530_101" in expName or "20240530_102" in expName:
            continue
        if expName in badExperiments:
            continue
        r, c = np.unravel_index(i, shape=(nRows, nCols))
        expData = np.array(featData[expName])
        proj = getMaxProj(expData)
        projZ = zscore(proj.copy(), nan_policy="omit")
        featGlob.append(proj.copy())
        # valMin, valMax = np.nanmin(proj), np.nanmax(proj)
        valMin, valMax = -2, 2
        projZ[edge == -1] = -100
        ax[r, c].imshow(np.rot90(projZ, 3), cmap=myCmap, vmin=valMin, vmax=valMax)
        ax[r, c].set_title(expName, size=4)
    for a in np.ravel(ax):
        a.axis("off")
    plt.savefig(f"{movieDir}/{featName}.png", dpi=600)
    plt.close()
