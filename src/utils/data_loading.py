import glob
import pickle
import numpy as np
from scipy.stats import zscore
from scipy.ndimage import label
from scipy.signal import savgol_filter as savgol
import h5py
from src.utils.fictrac import *
from src.utils.tracking import *
import os


def loadSupervoxelData(neuralDataPath, channel:int=1, getRaw:bool=False):
    # load data
    assert channel in [1 ,2], "invalid channel specified"
    neuralDataPath = glob.glob(dir + f"/**/channel_{channel}/**/supervoxel*.pkl", recursive=True)

    if len(neuralDataPath) > 0:
        neuralDataPath = neuralDataPath[0]
    else:
        raise FileNotFoundError("no supervoxel data found")
    print(neuralDataPath)

    with open(neuralDataPath, "rb") as handle:
        neuralData = pickle.load(handle)

    # create big matrix with neural data from all z slices
    sliceKeys = np.array(list(neuralData["data"].keys()))
    allNeuralData = []
    idxToROI = {}  # roi python index -> roi label in supervoxel map
    roiToIdx = {}  # roi label -> python index
    count = 0

    print(f"Grabbing supervoxel data (channel {channel})...")

    if getRaw:
        print("\ngetting raw (non-deconvolved) data...\n")

    for sliceKey in sliceKeys:
        if getRaw:
            sliceData = neuralData["data"][sliceKey]["dFF_raw"]
        else:
            sliceData = zscore(neuralData["data"][sliceKey]["dFF_cleaned"])
        labels = neuralData["data"][sliceKey]["labels"]
        allNeuralData.append(sliceData)

        for i in range(len(sliceData)):
            idxToROI[count] = int(labels[i])
            roiToIdx[int(labels[i])] = count
            count += 1

    allNeuralData = np.vstack(allNeuralData)
    allNeuralData[np.isnan(allNeuralData)] = 0
    print("Done!")
    return allNeuralData, idxToROI, roiToIdx


def loadFlyVRData(dir):
    print("Loading flyvr data...")
    fictracPath = glob.glob(
        dir + "/fictrac" + "/**/*[!daq,!video_server,!secondary,!tertiary].h5",
        recursive=True,
    )[0]
    vidPath = glob.glob(dir + "/fictrac" + "/**/*video_server*.h5", recursive=True)[0]
    daqPath = glob.glob(dir + "/fictrac" + "/**/*daq*.h5", recursive=True)[0]
    fictracData, vidData, daqData = (
        h5py.File(fictracPath, "r"),
        h5py.File(vidPath, "r"),
        h5py.File(daqPath, "r"),
    )
    daqSync = daqData["daq"]["chunk_synchronization_info"]  # sync between backends
    vidSync = vidData["video"]["synchronization_info"]
    fabs = daqSync[:, 0]  # fictrac at buffer sample
    favs = vidSync[:, 0]  # fictrac at video sample
    aabs = daqSync[:, 1]  # audio at buffer sample

    # load fictrac data
    speedDict, fps = loadFictracData(dir)

    # organize fictrac data
    fictracDataDict = {
        "fabs": fabs,
        "favs": favs,
        "aabs": aabs,
        "speedDict": speedDict,
        "fps": fps,
        "fictracData": fictracData,
        "vidData": vidData,
        "daqData": daqData,
    }
    print("Done!")
    return fictracDataDict


def loadCNNPredictions(dir, tapThresh=0.7, wingThresh=0.7):
    print("Loading CNN predictions...")
    tapFile = glob.glob(dir + "/**/predictions__tap.npz", recursive=True)
    wingFile = glob.glob(dir + "/**/predictions__wing.npz", recursive=True)

    if len(tapFile) == 0 or len(wingFile) == 0:
        print("no CNN predictions found", flush=True)
        return
    else:
        tapFile = tapFile[0]
        wingFile = wingFile[0]

    tapData, wingData = np.load(tapFile), np.load(wingFile)
    fps = 90  # fictrac fps

    # get behavior epochs
    wingBinary = np.where(
        np.max(wingData["predictionProbs"][:, 1:], axis=-1) > wingThresh, 1, 0
    )  # max. prob across wings
    wingBinaryL = np.where(wingData["predictionProbs"][:, 1] > wingThresh, 1, 0)
    wingBinaryR = np.where(wingData["predictionProbs"][:, 2] > wingThresh, 1, 0)
    tapBinary = np.where(tapData["predictionProbs"][:, -1] > tapThresh, 1, 0)
    wingEpochs, _ = label(wingBinary)
    wingEpochsL, _ = label(wingBinaryL)
    wingEpochsR, _ = label(wingBinaryR)
    wingEpochs, wingBinary = mergeBouts(wingEpochs)
    wingEpochsL, wingBinaryL = mergeBouts(wingEpochsL)
    wingEpochsR, wingBinaryR = mergeBouts(wingEpochsR)
    tapEpochs, _ = label(tapBinary)
    tapEpochs, tapBinary = mergeBouts(tapEpochs)

    # collect data
    cnnData = {
        "wingBinaryL": wingBinaryL,
        "wingBinaryR": wingBinaryR,
        "tapBinary": tapBinary,
    }
    print("Done!")
    return cnnData


# def getTrackingIndex(flyvrData, filterLen: int = 21):
#     vidData, favs, mRS, fps = (
#         flyvrData["vidData"],
#         flyvrData["favs"],
#         flyvrData["speedDict"]["rotational_speed"],
#         flyvrData["fps"],
#     )

#     # get desired time for resampling
#     desiredTimeFictrac = np.arange(len(mRS))  # fictrac samples for each video frame

#     # side-side female movement
#     stim = vidData["video"]["stimulus"]["actuator"]
#     sideSide = stim[:, 4]
#     sideSideResamp = np.interp(desiredTimeFictrac, favs, sideSide)

#     # chunk the data
#     avgDT = 1 / fps
#     nCycles = 1
#     nSamples = 120
#     stimChunks, peaks, trialCenters, trialTime, avgSamples = chunkTrackingData(
#         sideSideResamp, nCycles=nCycles, dt=avgDT, nSamples=nSamples, isBacknforth=True
#     )

#     # Chunk rotational speed
#     speedChunks, _, _, _, _ = chunkTrackingData(
#         mRS,
#         nCycles=nCycles,
#         peaks=peaks,
#         dt=avgDT,
#         nSamples=nSamples,
#         isSpeed=True,
#         isBacknforth=True,
#     )

#     trackingIndex, epochs, changePoints = computeTrackingIndex(speedChunks, stimChunks)
#     trackingIndex = savgol_filter(trackingIndex, window_length=filterLen, polyorder=2)

#     # Resample tracking index to align with fictrac samples
#     tiResampFic = np.interp(
#         desiredTimeFictrac, trialCenters, trackingIndex
#     )  # fictrac timebase

#     return tiResampFic
