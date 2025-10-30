import numpy as np
from scipy.signal import find_peaks, resample
import ruptures as rpt
from matplotlib import pyplot as plt
from scipy.ndimage import label
from scipy.stats import pearsonr
from scipy.signal import correlate, savgol_filter
import pandas as pd


def getMovingAverage(x, win):
    """Compute moving average"""
    cv = np.convolve(x, np.ones(win), "valid") / win
    return cv


def scale(x, r1, r2):
    """Scale input between range."""
    return (x - min(x)) / (max(x) - min(x)) * (r2 - r1) + r1


def resampleRows(x, nSamples):
    """Equalize number of samples in each row of input data.
    %% Inputs %%
    - x (array): input data (each row is a trial)
    - nSamples (int): number of samples in resampled row
    %% Outputs %%
    - resampled (array): resampled array with equal samples per row.
    """
    resampled = []

    for row in x:
        resampledRow = resample(row, nSamples)  # resapmled row
        resampled.append(resampledRow)

    resampled = np.array(resampled)

    return resampled


def chunkTrackingData(
    x, nCycles, dt, peaks=None, nSamples=1000, isSpeed=False, isBacknforth=False
):
    """Chunk vectorized data into trials.
    %% Inputs %%
    - x (list-like): vector data
    - nCycles (int): number of stimulus cycles per chunk
    - dT (float): average seconds/frame
    - peaks (list-like): pre-computed peaks used for chunking data
    - isSpeed (bool): flag indicating input is speed data
    - isBacknforth (bool): flag indicating if stimulus is backnforth (vs. grating)
    %% Outputs %%
    - chunkedData (array): chunked data
    - peaks (list): discovered peaks
    """
    # First, scale data (only if not speed)
    if not isSpeed:
        x = scale(x, -1, 1)

    # three peaks per cycle: high, low, high
    nPeaks = 2 * nCycles  # number of peaks *after* first peak -> three peaks

    # Identify peaks in data - peaks in second derivative (discontinuities)
    if peaks is None:
        posSignal = np.where(x > 0, 1, 0)
        negSignal = np.where(x < 0, 1, 0)
        peaksPos = find_peaks(posSignal, distance=100, height=0.1)[0]
        peaksNeg = find_peaks(negSignal, distance=100, height=0.1)[0]
        peaks = np.sort(list(peaksPos) + list(peaksNeg))
    # plt.plot(x);plt.scatter(peaks,x[peaks],c='r');plt.show()
    # import pdb; pdb.set_trace()

    # Chunk data by peaks
    xChunked = [
        x[peaks[ii * (nPeaks)] : peaks[nPeaks * (ii + 1)]]
        for ii, _ in enumerate(peaks)
        if ii * (nPeaks) + nPeaks < len(peaks)
    ]

    # get center of each trial
    trialCenters = np.array(
        [
            np.mean(peaks[ii * (nPeaks) : nPeaks * (ii + 1)])
            for ii, _ in enumerate(peaks)
            if ii * (nPeaks) + nPeaks < len(peaks)
        ]
    )

    # Compute average time per trial
    avgTrialTime = dt * np.mean(
        [
            peaks[ii * (nPeaks) + nPeaks] - peaks[ii * (nPeaks)]
            for ii, _ in enumerate(peaks)
            if ii * (nPeaks) + nPeaks < len(peaks)
        ]
    )

    # Compute average number of samples per trial
    avgSamples = np.mean(
        [
            peaks[ii * (nPeaks) + nPeaks] - peaks[ii * (nPeaks)]
            for ii, _ in enumerate(peaks)
            if ii * (nPeaks) + nPeaks < len(peaks)
        ]
    )

    # Resample chunked data to equalize samples per row
    try:
        xResampled = resampleRows(xChunked, nSamples=nSamples)
    except:
        return None, None, None, None, None
    return xResampled, peaks, trialCenters, avgTrialTime, avgSamples


def detect_change_points(data, penalty_value=10):
    # Create a Pelt model for change point detection
    model = rpt.Pelt(model="rbf").fit(data)

    # Find the change points
    change_points = model.predict(pen=penalty_value)

    return change_points


def plot_epochs(data, change_points):
    fig, ax = plt.subplots(figsize=(15, 7))

    # Plot the original data
    im = ax.imshow(data, aspect="auto", cmap="coolwarm")
    plt.colorbar(im, label="Rotational speed (rad/frame)")

    # Plot horizontal lines at change points
    for cp in change_points[
        :-1
    ]:  # Exclude the last point which is the end of the series
        ax.axhline(y=cp, color="g", linestyle="--")

    ax.set_title("Change Point Detection Results")
    plt.tight_layout()
    plt.show()


def computeTrackingIndex(speedChunks0, stimChunks0):

    # make working copies
    speedChunks = abs(speedChunks0.copy())
    stimChunks = abs(stimChunks0.copy())

    # compute average stim chunk
    muStim = np.mean(stimChunks, axis=0)

    # Detect change points
    change_points = detect_change_points(speedChunks, penalty_value=5)

    # % get tracking signal during each epoch
    start = 0  # start of epoch
    epochs = []

    for epochEnd in change_points:
        epoch = np.arange(start, epochEnd, 1)
        epochs.append(epoch)
        start = epochEnd  # update epoch

    # get signal during each epoch
    epochSignals = [speedChunks[epoch] for epoch in epochs]
    muSignals = np.array([np.mean(x, axis=0) for x in epochSignals])
    epochTracking = []

    # compute speed correlations
    corrs = [
        np.array([pearsonr(epoch[ii], mu)[0] for ii in range(len(epoch))])
        for (epoch, mu) in zip(epochSignals, muSignals)
    ]

    # get normalized speeds
    speedNorm = np.mean(abs(speedChunks), axis=-1)
    speedNorm /= np.max(speedNorm)

    # compute TI
    for i in range(len(epochSignals)):
        corr = corrs[i]
        epochTracking.extend(corr)

    rollLen = 6
    epochTracking = np.pad(epochTracking, (rollLen - 1, 0), mode="edge")
    df = pd.DataFrame(np.array(epochTracking))
    trackingFiltered = np.squeeze(df.rolling(rollLen).mean())
    trackingFiltered = trackingFiltered[rollLen:]
    # plt.plot(trackingFiltered); plt.show()

    return trackingFiltered, epochs, change_points


def getTrackingIndex(flyvrData, filterLen: int = 21, returnEpochs=False):

    vidData, favs, mRS, fps = (
        flyvrData["vidData"],
        flyvrData["favs"],
        flyvrData["speedDict"]["rotational_speed"],
        flyvrData["fps"],
    )

    # get desired time for resampling
    desiredTimeFictrac = np.arange(len(mRS))  # fictrac samples for each video frame

    # side-side female movement
    stim = vidData["video"]["stimulus"]["actuator"]
    sideSide = stim[:, 4]
    sideSideResamp = np.interp(desiredTimeFictrac, favs, sideSide)

    # chunk the data
    avgDT = 1 / fps
    nCycles = 2
    nSamples = 120
    stimChunks, peaks, trialCenters, trialTime, avgSamples = chunkTrackingData(
        sideSideResamp, nCycles=nCycles, dt=avgDT, nSamples=nSamples, isBacknforth=True
    )

    # Chunk rotational speed
    speedChunks, _, _, _, _ = chunkTrackingData(
        mRS,
        nCycles=nCycles,
        peaks=peaks,
        dt=avgDT,
        nSamples=nSamples,
        isSpeed=True,
        isBacknforth=True,
    )
    trackingIndex, epochs, changePoints = computeTrackingIndex(speedChunks, stimChunks)

    # Resample tracking index to align with fictrac samples
    minLen = min(len(trialCenters), len(trackingIndex))

    tiResampFic = np.interp(
        desiredTimeFictrac, trialCenters[:minLen], trackingIndex[:minLen]
    )  # fictrac timebase

    if not returnEpochs:
        return tiResampFic
    else:
        changePointsFic = [
            int(trialCenters[x]) for x in changePoints[:-1]
        ]  # last change point is end of experiment
        return tiResampFic, changePointsFic


def showTracking(flyvrData, returnFig=False):

    vidData, favs, mRS, fps = (
        flyvrData["vidData"],
        flyvrData["favs"],
        flyvrData["speedDict"]["rotational_speed"],
        flyvrData["fps"],
    )

    # get desired time for resampling
    desiredTimeFictrac = np.arange(len(mRS))  # fictrac samples for each video frame

    # side-side female movement
    stim = vidData["video"]["stimulus"]["actuator"]
    sideSide = stim[:, 4]
    sideSideResamp = np.interp(desiredTimeFictrac, favs, sideSide)

    # chunk the data
    avgDT = 1 / fps
    nCycles = 2
    nSamples = 120
    stimChunks, peaks, trialCenters, trialTime, avgSamples = chunkTrackingData(
        sideSideResamp, nCycles=nCycles, dt=avgDT, nSamples=nSamples, isBacknforth=True
    )

    # Chunk rotational speed
    speedChunks, _, _, _, _ = chunkTrackingData(
        mRS,
        nCycles=nCycles,
        peaks=peaks,
        dt=avgDT,
        nSamples=nSamples,
        isSpeed=True,
        isBacknforth=True,
    )
    trackingIndex, epochs, changePoints = computeTrackingIndex(speedChunks, stimChunks)

    # get trial info
    trialIDs = np.arange(len(trackingIndex))
    trialTimestamps = (trialTime * trialIDs / 60).astype(int)
    uniqueTimestamps = [
        np.ravel(np.argwhere(trialTimestamps == t)[0])
        for t in np.sort(np.unique(trialTimestamps))
    ]
    uniqueTimestamps = np.ravel(uniqueTimestamps)

    # make plot
    import matplotlib.colors as colors
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import seaborn as sns

    minTrack = np.min(trackingIndex)
    maxTrack = np.max(trackingIndex)

    fig = plt.figure(constrained_layout=True)
    spec = fig.add_gridspec(
        10, 4, width_ratios=[3, 0.05, 0.2, 1]  # tweak these numbers as you like
    )
    ax0 = fig.add_subplot(spec[:2, :2])
    ax1 = fig.add_subplot(spec[2:, :2])
    ax2 = fig.add_subplot(spec[2:, 3])
    ax0.plot(np.mean(stimChunks, axis=0), color="k")
    ax0.axis("off")
    maxSpeed = np.max(abs(speedChunks))
    vmin, vmax = -maxSpeed, maxSpeed
    nRows, nCols = np.shape(speedChunks)
    im = ax1.imshow(
        speedChunks,
        aspect="auto",
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        extent=[0, nCols, 0, nRows],  # x in [0, nCols], y in [0, nRows]
    )
    ax1.set_xticks([0, nSamples])
    ax1.set_xticklabels([0, f"{trialTime:.0f}"], size=8)
    ax1.set_yticks(trialIDs[uniqueTimestamps])
    ax1.set_yticklabels(trialTimestamps[uniqueTimestamps[::-1]].astype(int), size=8)
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("time (min)")
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(
        im, cax=cax, orientation="vertical", 
    )
    cbar.ax.tick_params(labelsize=8) 
    cbar.set_label(label="rotational speed (rad/s)", labelpad=-10)

    cbar.ax.yaxis.set_ticks_position("right")

    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{int(np.round(vmin))}", f"{int(np.round(vmax))}"])
    ax2.plot(trackingIndex, trialIDs, color="k")
    ax2.set_xticks([0, maxTrack])
    ax2.set_xticklabels(["0", f"{np.round(maxTrack, 1)}"], size=8)
    sns.despine(ax=ax2, top=True, bottom=False, left=True)
    ax2.set_xlim(minTrack, np.max(trackingIndex))
    ax2.set_ylim(0, nRows)
    ax2.invert_yaxis()
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel("tracking index (a.u.)")

    if returnFig:
        return fig
    plt.show()


def getBehavioralEpochs(flyvrData, filterLen: int = 21):

    vidData, favs, mRS, fps = (
        flyvrData["vidData"],
        flyvrData["favs"],
        flyvrData["speedDict"]["rotational_speed"],
        flyvrData["fps"],
    )

    # get desired time for resampling
    desiredTimeFictrac = np.arange(len(mRS))  # fictrac samples for each video frame

    # side-side female movement
    stim = vidData["video"]["stimulus"]["actuator"]
    sideSide = stim[:, 4]
    sideSideResamp = np.interp(desiredTimeFictrac, favs, sideSide)

    # chunk the data
    avgDT = 1 / fps
    nCycles = 1
    nSamples = 120
    stimChunks, peaks, trialCenters, trialTime, avgSamples = chunkTrackingData(
        sideSideResamp, nCycles=nCycles, dt=avgDT, nSamples=nSamples, isBacknforth=True
    )

    # Chunk rotational speed
    speedChunks, _, _, _, _ = chunkTrackingData(
        mRS,
        nCycles=nCycles,
        peaks=peaks,
        dt=avgDT,
        nSamples=nSamples,
        isSpeed=True,
        isBacknforth=True,
    )

    _, epochs, changePoints = computeTrackingIndex(speedChunks, stimChunks)

    return epochs, changePoints, trialCenters


def filterIslands(x, minLen=2):
    """filter out islands in thresholded data"""
    xLabels, _ = label(x)
    labelVals = np.unique(xLabels[xLabels > 0])
    for l in labelVals:
        whereVal = np.argwhere(xLabels == l)
        if len(whereVal) < minLen:
            x[whereVal] = 0
    return x


def getEpochBoundaries(labelData):
    boundaries = []
    for l in np.unique(labelData[labelData > 0]):
        whereLabel = np.argwhere(labelData == l)
        epochStart, epochEnd = whereLabel[0][0], whereLabel[-1][0]
        boundaries.append((epochStart, epochEnd))
    return boundaries
