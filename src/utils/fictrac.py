# author: Max Aragon
# description: Utility functions fictrac-related analysis
# tags: fictrac

import glob
import h5py
import numpy as np
from bisect import bisect_left
import pandas as pd
from src.utils.neural_data import *
from scipy.signal import savgol_filter as savgol


def loadFictracData(directory: str):
    """Load fictrac data from directory.
    %% Inputs %%
    - directory: parent directory containing fictrac folder
    %% Outputs %%
    - scaledRS: z-scored rotational speed calculated for each fictrac frame
    - rs: rotational speed
    - fps: average fictrac framerate for experiment
    """
    # Load fictrac data
    fictracPath = glob.glob(
        directory + "/fictrac" + "/**/*[!daq,!video_server,!secondary,!tertiary].h5",
        recursive=True,
    )
    if len(fictracPath) == 0:
        raise FileNotFoundError(f"no fictrac h5 dataset found in directory {directory}")

    fictracData = h5py.File(fictracPath[0], "r")

    # Get rotational speed
    filterLen = 45  # approximately 500 ms
    timestamps = fictracData["fictrac"]["output"][:, 21]  # fictrac timestamps
    fps = (np.mean(np.diff(timestamps)) / 1000) ** -1  # frames per second
    rs = savgol(
        fps * fictracData["fictrac"]["output"][:, 2],
        window_length=filterLen,
        polyorder=3,
    )  # rotational speed
    fs = savgol(
        fps * fictracData["fictrac"]["output"][:, 3],
        window_length=filterLen,
        polyorder=3,
    )  # forward speed
    ls = savgol(
        fps * fictracData["fictrac"]["output"][:, 4],
        window_length=filterLen,
        polyorder=3,
    )  # lateral speed
    speedDict = {"rotational_speed": rs, "forward_speed": fs, "lateral_speed": ls}
    return speedDict, fps


def getVideoData(dataDir):
    """get video data -> resample to fictrac time base
    %% Inputs %%
    - dataDir: directory containing fictrac data
    %% Outputs %%
    vidDataDict: contains vid data resampled to fictrac timebase
    """
    # load daq data
    try:
        fictracPath = dataDir + "/fictrac"
        vidServerPath = glob.glob(f"{fictracPath}/*video_server.h5")[0]
        daqPath = glob.glob(f"{fictracPath}/*daq.h5")[0]
        vid = h5py.File(vidServerPath, "r")  # vid data
        daq = h5py.File(daqPath, "r")  # daq data
        dset = h5py.File(
            glob.glob(f"{fictracPath}/*[!video_server,!daq,!secondary,!tertiary].h5")[0]
        )
    except:
        print("dataset not found", flush=True)
        return None

    # Get synchronization information for fictrac and stimulus backends
    sync = daq["daq"]["input"]["synchronization_info"]
    fictracSync = sync[:, 0]  # fictrac frames corresponding to buffer chunks
    stimSync = sync[:, 1]  # daq samples (audio rate) corresponding to buffer chunks
    vidSync = vid["video"][
        "synchronization_info"
    ]  # data stream samples corresponding to video frames (psychopy)

    # Get average fictrac fps
    deltaTimestamps = dset["fictrac"]["output"][:, 21]
    fps = (np.mean(np.diff(deltaTimestamps)) / 1000) ** -1  # frames per second

    # Get visual stimulus features
    desiredTime = np.arange(
        len(deltaTimestamps)
    )  # want resampled video frames to match each fictrac frame

    # If contrast stimulus is longer than backnforth, this is a widefield contrast experiment. Use widefield contrast data from h5 file
    vidStim = vid["video"]["stimulus"]  # vid stim fields
    vidStimDset = vid["video"]["stimulus"]["actuator"]  # default to backnforth data

    # Get visual features from female stimulus
    mfDist = vidStimDset[:, 3]  # male-female distance
    mfDist = (
        np.max(mfDist) - mfDist
    )  # from 0-25 originally, where 25 is closest. make this a real distance now
    mfAng = vidStimDset[:, 4]  # male-female angle

    # Resample visual features to fictrac time base
    originalTimeVid = vidSync[:, 0]  # fictrac frame for each video frame
    try:
        assert len(originalTimeVid) == len(mfDist)
    except:
        return None
    mfDist = np.interp(desiredTime, originalTimeVid, mfDist)
    mfAng = np.interp(desiredTime, originalTimeVid, mfAng)

    return {"mfDist": mfDist, "mfAng": mfAng}


def getClosestBufferSampToAudio(audioSamples: list, aabs: list):
    """given an audio sample, find the closest matching buffer sample
    % Inputs
    - audioSamples: query samples
    - aabs: audio sample at buffer sample"""
    # audioSamples,aabs = list(audioSamples),list(aabs)
    # matchingBufferSample = [aabs.index(min(aabs, key = lambda aabs : abs(audioSamp - aabs))) \
    #                         for audioSamp in audioSamples]
    # import pdb; pdb.set_trace()

    audioSamples, aabs = list(audioSamples), list(aabs)
    matchingBufferSample = [binarySearch(aabs, audioSamp) for audioSamp in audioSamples]
    return matchingBufferSample


def binarySearch(source, query):
    idx = bisect_left(source, query)

    if idx == 0:
        return 0
    if idx == len(source):
        return len(source) - 1
    before = source[idx - 1]
    after = source[idx]

    if after - query < query - before:
        return idx
    else:
        return idx - 1


def mergeBouts(bouts: list, fps: int = 90, minBoutDist: int = 200, minBoutLen:int = 200):
    """merge bouts that are close together in time
    % Inputs
    - bouts: bout ID at each frame (from scipy ndimage.label)
    - fps: fictrac frames per second
    - minBoutDist: minimum time between distinct song bouts (ms)
    - minBoutLen: minimum bout duration (ms)
    % Outputs
    - mergedBouts: merged song bouts
    """
    if max(bouts) <= 1:
        return np.zeros_like(bouts), np.zeros_like(bouts)
    bouts = np.array(bouts)
    mergedBouts = np.zeros(len(bouts))  # contains merged song bouts
    boutIDs = np.unique(np.sort(bouts[bouts > 0]))  # labels for bouts
    currentBout = 1  # current bout we're at
    minBoutDist = int(fps * minBoutDist / 1000)  # min samples between bouts

    # iterate through each bout and compare to previous bout's end time
    for ii, id in enumerate(boutIDs):
        # nothing to compare to for first bout: just add it in
        if ii == 0:
            boutStart = np.squeeze(np.argwhere(bouts == id)[0])
            boutEnd = np.squeeze(np.argwhere(bouts == id)[-1])
            mergedBouts[boutStart : boutEnd + 1] = id
            continue

        # for subsequent bouts, need to see where this bout starts
        # relative to the end of the previous bout
        prevBoutStart = np.squeeze(
            np.argwhere(mergedBouts == currentBout)[0]
        )  # first frame of previous bout
        prevBoutEnd = np.squeeze(
            np.argwhere(mergedBouts == currentBout)[-1]
        )  # last frame of previous bout
        newBoutStart = np.squeeze(
            np.argwhere(bouts == id)[0]
        )  # first frame of this bout
        newBoutEnd = np.squeeze(np.argwhere(bouts == id)[-1])  # last from of this bout
        dt = (
            newBoutStart - prevBoutEnd
        )  # nFrames between last bout end and this bout start

        # if we're within the merge threshold, all frames between the previous bout's start
        # and this bouts end should be labeled as the same bout
        if dt < minBoutDist:
            mergedBouts[prevBoutStart : newBoutEnd + 1] = currentBout
        # if we're outside the merge threshold, this bout should be considered distinct
        # from the previous bout
        else:
            currentBout += 1  # this is a distinct bout, so we need to update the bout index.
            mergedBouts[newBoutStart : newBoutEnd + 1] = currentBout

    # now filter out short bouts 
    for l in np.unique(mergedBouts[mergedBouts > 0]):
        boutFrames = mergedBouts == l
        if sum(boutFrames) < minBoutLen * fps/1000:
            mergedBouts[boutFrames] = 0  # set to zero

    return mergedBouts, np.where(mergedBouts > 0, 1, 0)


def getFictracFrameForVideo(fictracDirectory: str):
    """extract fictrac frame for each video frame.
    % Inputs
    - dataPath: path to fictrac directory
    % Outputs
    - ficFramesAtVideo (list): fictrac sample at each camera frame
    """

    vidLogFile = glob.glob(fictracDirectory + "/test_out-vidLog*.txt")

    if len(vidLogFile) > 0:
        vidLogFile = vidLogFile[0]
    else:
        raise FileNotFoundError("no video log file found for primary camera.")

    # read the text file
    ficFramesAtVideo = np.squeeze(np.array(pd.read_csv(vidLogFile, sep=",")))

    return ficFramesAtVideo


def getFictracSampsAtVolume(flyvrData, nSlices=40):
    daqData = flyvrData["daqData"]
    aabs = flyvrData["aabs"]  # audio sample at buffer sample
    fabs = flyvrData["fabs"]  # fictrac sample at buffer sample
    print(nSlices)
    vaas, volStarts = getVAAS(daqData, nSlices=nSlices)  # volume at audio sample
    print(volStarts)
    bufferSamples = getClosestBufferSampToAudio(
        audioSamples=volStarts, aabs=aabs
    )  # buffer sample closest to mirror peaks
    fictracSampsAtVolume = fabs[bufferSamples]  # fictrac sample at imaging volume
    return fictracSampsAtVolume
