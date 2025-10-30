# % imports
import pdb
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.ndimage import label
from sklearn.decomposition import PCA
import glob
import os
import argparse
import matplotlib
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter
from skimage.filters import threshold_triangle as triangle
from sklearn.feature_extraction.image import grid_to_graph
from sklearn.linear_model import LinearRegression
from pathlib import Path
import warnings

# matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore, pearsonr
from scipy.ndimage import binary_erosion, binary_dilation
from pathos.multiprocessing import ProcessingPool
import sys
import psutil
import time
from skimage import segmentation as segment
from sklearn.mixture import GaussianMixture
from skimage.filters import sobel
from scipy import ndimage as ndi
from scipy.signal import resample, find_peaks
from scipy.linalg import toeplitz
from oasis.functions import deconvolve
import pickle
from skimage.exposure import equalize_hist

# % parse args
parser = argparse.ArgumentParser(description="""segment neural data""")
parser.add_argument("--fname", type=str, required=True, help="path to mmap data")
args = parser.parse_args()


# % defs
def segmentImage(
    imPath: str,
    imShape: tuple,
    zIndex: int,
    whereEnd: int,
    clusterMap: list,
    nGaussians: int,
    isSparse: bool = False,
    segmentationMask: np.array = None,
):
    """args
    - segmentationMask: optional binary mask. if provided, skip segmentation step.
    """
    # -----------------------------------------------
    # first step: identify background and brain
    # -----------------------------------------------
    Z, T, X, Y = imShape

    # read data
    images = np.memmap(imPath, shape=imShape, mode="r+", dtype=np.float32)  # Z, T, X, Y
    images = np.transpose(images, (1, -2, -1, 0))  # T,X,Y,Z
    zSlice = images[:, :, :, zIndex]
    image = np.mean(zSlice, axis=0)

    # get brainMask
    brainMask = segmentImageBifrost(image)

    # apply mask
    clusterMap *= brainMask

    return clusterMap


def segmentImageBifrost(brain):
    ### Blur brain and mask small values ###
    brain_copy = brain.copy().astype("float32")
    brain_copy = equalize_hist(brain_copy)
    brain_copy = gaussian_filter(brain_copy, sigma=5)
    threshold = triangle(brain_copy)
    brain_copy[np.where(brain_copy < threshold / 2)] = 0

    ### Remove blobs outside contiguous brain ###
    labels, _ = label(brain_copy)
    brain_label = np.bincount(labels.flatten())[1:].argmax() + 1
    brainMask = brain.copy().astype("float32")
    brainMask[np.where(labels != brain_label)] = 0
    brainMask[np.where(labels == brain_label)] = 1

    return brainMask


def clusterPixelsWithConnectivity(
    data: list,
    nClusters: int,
):
    """
    Clusters pixels in calcium imaging data based on the similarity of their time traces
    using hierarchical clustering with a connectivity constraint.

    Args:
        data (numpy.ndarray): Calcium imaging data with shape (time, xResolution, yResolution).
        nClusters (float): number of clusters.
    Returns:
        numpy.ndarray: An array of cluster labels with shape (xResolution, yResolution).
    """
    import numpy as np
    from sklearn.cluster import AgglomerativeClustering

    clusteredTiles = []
    connectivityMatrix = grid_to_graph(*data[0][0].shape)
    # iterate over tiles

    for i, tile in enumerate(data):
        # get connectivity matrix for each pixel
        # if i == 0:
        #     connectivityMatrix = makeConnectivityMatrix(tile, connectivity=1)
        # connectivityMatrix = None

        # Reshape the data to a 2D array of pixels and their time traces
        tile = tile.copy()
        numPixels = tile.shape[1] * tile.shape[2]
        pixelTraces = tile.reshape((tile.shape[0], numPixels)).T
        pixelTraces = zscore(pixelTraces, axis=1, nan_policy="omit")
        pixelTraces[np.isnan(pixelTraces)] = 0

        # Perform hierarchical clustering with connectivity constraint
        clustering = AgglomerativeClustering(
            n_clusters=nClusters, connectivity=connectivityMatrix, linkage="ward"
        )
        clusterLabels = clustering.fit_predict(pixelTraces).astype(float)
        clusterLabels += i * nClusters  # distinguish clusters between tiles

        # Reshape the cluster labels back to the original data shape
        clusterLabels = clusterLabels.reshape((tile.shape[1], tile.shape[2]))

        # append cluster labels
        clusteredTiles.append(clusterLabels)

    clusteredTiles = np.array(clusteredTiles)
    return clusteredTiles


def getSupervoxels(
    imPath: list,
    zIndex: int,
    imDim: tuple,
    whereEnd: int,
    tileSize: int = 8,
    clustersPerPatch: int = 20,
    brainMask=None,
):
    Z, T, X, Y = imDim

    images = np.memmap(imPath, shape=imDim, mode="r+", dtype=np.float32)  # Z, T, X, Y
    images = np.transpose(images, (1, -2, -1, 0))  # T,X,Y,Z

    # get z slice
    _, _, _, nZ = images.shape
    superVoxels = []
    zSlice = np.array(images[:, :, :, zIndex])

    if brainMask is not None:
        zSliceMask = brainMask[:, :, zIndex]
        zSlice = zSlice * zSliceMask

    # Tile zSlice
    if tileSize is None:
        tilesProcess = zSlice[None, :, :, :]

    else:
        tilesProcess = np.array(
            [tileImage(np.ascontiguousarray(x), tileSize) for x in zSlice]
        )  # (T, nTiles, xRes, yRes)
        tilesProcess = np.moveaxis(tilesProcess, 0, 1)  # (nTiles, T, xRes, yRes)

    clusteredTiles = clusterPixelsWithConnectivity(
        tilesProcess, nClusters=clustersPerPatch
    )

    # merge the tiles
    clusteredTiles = clusteredTiles[None, :, :, :]
    nRows, nCols = int(X / tileSize), int(Y / tileSize)
    clusteredTiles = [
        [np.hstack(x[ii * nCols : (ii + 1) * nCols]) for ii in range(nRows)]
        for x in clusteredTiles
    ]  # merge horizontally
    clusteredTiles = np.squeeze([np.vstack(x) for x in clusteredTiles])
    superVoxels = np.array(clusteredTiles)

    # add offset to supervoxels: need unique indices for each slice.
    # maximum num. supervoxels is clustersPerPatch * nPatches
    nPatches = nRows * nCols  # number of patches for segmentation
    offset = nPatches * clustersPerPatch * zIndex
    superVoxels += offset
    superVoxels *= zSliceMask  # apply final mask to supervoxels

    return superVoxels, offset


def getBackgroundSignal(imPath: str, zIndex: int, imDim: tuple, brainMask):

    # load data
    Z, T, X, Y = imDim
    images = np.memmap(imPath, shape=imDim, mode="r+", dtype=np.float32)  # Z, T, X, Y
    imageSlice = images[zIndex]  # T, X, Y

    # get background mask
    brainMaskInverse = ~brainMask.copy().astype(bool)
    brainMaskInvSlice = brainMaskInverse[:, :, zIndex]

    # get background signal
    bkg = imageSlice[:, brainMaskInvSlice]
    muBkg = np.nanmean(bkg, axis=1)

    return muBkg


def getBackgroundPixels(imPath: str, zIndex: int, imDim: tuple, brainMask):

    # load data
    Z, T, X, Y = imDim
    images = np.memmap(imPath, shape=imDim, mode="r+", dtype=np.float32)  # Z, T, X, Y
    imageSlice = images[zIndex]  # T, X, Y

    # get background mask
    brainMaskInverse = ~brainMask.copy().astype(bool)
    brainMaskInvSlice = brainMaskInverse[:, :, zIndex]

    return brainMaskInvSlice


def tileImage(im, tileSize):
    """Tile image."""

    def getNewShape(im, tileSize):
        xRes, yRes = im.shape
        newShape = (int(xRes / tileSize), int(yRes / tileSize), tileSize, tileSize)
        return newShape

    def getNewStrides(im, tileSize):
        xRes, yRes = im.shape
        byteSize = im.itemsize
        newStrides = (
            yRes * tileSize * byteSize,
            tileSize * byteSize,
            yRes * byteSize,
            byteSize,
        )
        return newStrides

    # Make sure image can be evenly tiled
    imShape = im.shape

    # Generate tiles
    newShape = getNewShape(im, tileSize)
    newStrides = getNewStrides(im, tileSize)
    tiles = np.lib.stride_tricks.as_strided(im, shape=newShape, strides=newStrides)
    tiles = np.reshape(tiles, (-1, tileSize, tileSize))

    return tiles


def calculateDff(calciumSignal, windowSize=100, percentile: int = 20, bkgOnly=False):
    """
    Calculates the dF/F calcium signal with baseline correction using a sliding window.

    Args:
        calciumSignal (numpy.ndarray): 1D array of calcium signal values.
        windowSize (int): Size of the sliding window for baseline estimation (default: 100).

    Returns:
        numpy.ndarray: 1D array of dF/F values.
    """
    # Pad the calcium signal at the beginning and end to preserve shape after dF/F calculation
    paddedSignal = np.pad(
        calciumSignal, (windowSize // 2, windowSize - 1 - windowSize // 2), mode="edge"
    )

    # Initialize the baseline and dF/F arrays
    baseline = np.zeros_like(calciumSignal)
    dff = np.zeros_like(calciumSignal)

    # Iterate over each time point in the calcium signal
    for i in range(len(calciumSignal)):
        # Extract the window around the current time point
        window = paddedSignal[i : i + windowSize]

        # Estimate the baseline using a percentile-based method (e.g., 10th percentile)
        baseline[i] = np.percentile(window, percentile)

        # Calculate the dF/F value for the current time point
        if bkgOnly:
            # dff[i] = (calciumSignal[i])/baseline[i]
            dff[i] = calciumSignal[i] - baseline[i]
        else:
            dff[i] = (calciumSignal[i] - baseline[i]) / baseline[i]

    return dff


def regressOutMotion(green, red):
    """
    Regresses out motion from red channel.

    Args:
        green: green signal
        red: red signal

    Returns:
        mcSignal: motion-corrected signal
    """
    red = np.concatenate((np.ones(len(red))[:, None], red), axis=1)
    try:
        betas = np.linalg.pinv(red.T @ red) @ red.T @ green
        greenMotion = red @ betas
        mcSignal = green - greenMotion

    except:
        mcSignal = green
        greenMotion = np.zeros_like(mcSignal)
        warnings.warn("SVD did not converge for pseudoinverse calculation")

    return mcSignal, greenMotion


def getDeconvolvedTrace(
    fname: str,
    imDim: tuple,
    zIndex: int,
    whereEnd: int,
    labelIndex: int,
    clusterMap: list,
    bkgGreen: list,
    bkgRed: list,
):
    """get calcium signals for each cluster

    Args:
    - fname: path to memmaped calcium imaging data
    - imDim: imaging data dimensions
    - zIndex: z slice index to process
    - whereEnd: where to end the recording if something bad happened
    - labelIndex: which cluster label to process
    - clusterMap: supervoxel map
    - brainMask: binary brain mask

    Returns:
    - deconvolved: deconvolved calcium transients for this cluster
    """
    Z, T, X, Y = imDim

    # read memory mapped data
    images = np.memmap(fname, shape=imDim, mode="r+", dtype=np.float32)  # Z, T, X, Y
    images = np.transpose(images, (1, -2, -1, 0))  # T,X,Y,Z

    # get calcium signals for this z slice
    calciumSignals = images[:, :, :, zIndex]

    # get average signal within supervoxel
    clusterMask = clusterMap == labelIndex
    clusterMean = np.mean(calciumSignals[:, clusterMask], axis=1)

    # remove background signal
    clusterMean -= bkgGreen

    # get red signal
    baseDir = str(Path(fname).parents[1])
    fnameRed = glob.glob(baseDir + "/**/*channel_2*.mmap")
    fnameRed = fnameRed[0]
    imagesRed = np.memmap(
        fnameRed, shape=imDim, mode="r+", dtype=np.float32
    )  # Z, T, X Y
    imagesRed = np.transpose(imagesRed, (1, -2, -1, 0))  # T,X,Y,Z
    redSignals = imagesRed[:, :, :, zIndex]
    clusterMeanRed = np.mean(redSignals[:, clusterMask], axis=1)
    clusterMeanRed -= bkgRed

    # get raw dff
    dffGreen = calculateDff(clusterMean, windowSize=100, percentile=50, bkgOnly=True)
    dffRed = calculateDff(clusterMeanRed, windowSize=100, percentile=50, bkgOnly=True)

    # mean-subtract
    dffGreen -= np.mean(dffGreen)
    dffRed -= np.mean(dffRed)

    # fit least squares. regress green signal onto red
    dffCleaned, _ = regressOutMotion(dffGreen[:, None], dffRed[:, None])

    # calculate terms for exclusion criterion
    numerator = np.sum(dffCleaned**2)
    denominator = np.sum(dffGreen**2)

    return [numerator, denominator]

# -----------------------------------------------
# % get number of available cpus for this task
# -----------------------------------------------
# ncpus = len(os.sched_getaffinity(0))
ncpus = 12
print(f"found {ncpus} cpus", flush=True)

# -----------------------
# % load memmaped data
# -----------------------

directory = os.path.dirname(args.fname)

# get data dimensions from filename
basename = os.path.basename(args.fname)
split = np.array(basename.split("__")[1].split("_"))
X, Y, Z = tuple(split[[1, 3, 5]].astype(int))
T = int(split[-2])
imDim = (Z, T, X, Y)

# make brain mask
images = np.memmap(args.fname, shape=imDim, mode="r+", dtype=np.float32)  # Z, T, X, Y
images = np.transpose(images, (1, -2, -1, 0))  # T,X,Y,Z
brainMask = segmentImageBifrost(np.mean(images, axis=0))  # t-projection

# ------------------------------
# % iterate through each z slice
# ------------------------------

# define parameters
tileSize = 128  # size of each tile
clustersPerPatch = 1000  # number of clusters per tile
corrThresh = 0.95  # threshold for merging supervoxels

# initialize dF/F dictionary
dffDict = {
    "data": {},
    "parameters": {
        "tile_size": tileSize,
        "clusters_per_patch": clustersPerPatch,
        "merge_threshold": corrThresh,
    },
}

# iterate over each z slice
allRes = []

for zIndex in tqdm(range(Z)):
    dffDict["data"][f"zSlice_{zIndex:02d}"] = {}

    print(f"processing slice {zIndex}...")

    # ---------------------
    # % cluster the data
    # ---------------------

    # get supervoxels for this slice
    supervoxels, offset = getSupervoxels(
        imPath=args.fname,
        imDim=imDim,
        tileSize=tileSize,
        clustersPerPatch=clustersPerPatch,
        whereEnd=None,
        zIndex=zIndex,
        brainMask=brainMask,
    )

    # ------------------------------------------
    # % get background red and green signals
    # ------------------------------------------
    baseDir = str(Path(args.fname).parents[1])
    fnameRed = glob.glob(baseDir + "/**/*channel_2*.mmap")[0]
    bkgGreen = getBackgroundSignal(
        args.fname, zIndex=zIndex, imDim=imDim, brainMask=brainMask
    )
    bkgRed = getBackgroundSignal(
        fnameRed, zIndex=zIndex, imDim=imDim, brainMask=brainMask
    )

    # ------------------------------------------------
    # % get calcium traces from segmented cluster maps
    # ------------------------------------------------
    pool = ProcessingPool(nodes=ncpus)

    # get unique non-zero supervoxel labels
    uniqueLabels = np.unique(supervoxels[supervoxels != 0])
    print(f"found {len(uniqueLabels)} supervoxels")

    # deconvolve calcium data

    # test = getDeconvolvedTrace(
    #     fname=args.fname,
    #     imDim=imDim,
    #     zIndex=zIndex,
    #     labelIndex=uniqueLabels[4],
    #     whereEnd=None,
    #     clusterMap=supervoxels,
    #     bkgGreen=bkgGreen,
    #     bkgRed=bkgRed,
    # )

    getDeconvolvedPar = lambda l: getDeconvolvedTrace(
        fname=args.fname,
        imDim=imDim,
        zIndex=zIndex,
        labelIndex=l,
        whereEnd=None,
        clusterMap=supervoxels,
        bkgGreen=bkgGreen,
        bkgRed=bkgRed,
    )
    res = pool.map(getDeconvolvedPar, uniqueLabels)
    allRes += res
    
    # clean up the pool
    pool.close()
    pool.clear()

out = np.sum(np.array(allRes)[:, 0])/np.sum(np.array(allRes)[:, 1])
print(f"metric: {out}", flush=True)





