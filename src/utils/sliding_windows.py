import numpy as np
from tqdm import tqdm


def chunkActivity(input_list, row_length, k):
    """
    Chunk the activity data into rows of specified length, with each subsequent row shifted by 'k' relative to the previous one.

    Parameters:
    input_list (list): The list to be chunked.
    row_length (int): The length of each row.
    k (int): The stride or shift for each subsequent row.

    Returns:
    list: A list of rows, where each row is a list shifted accordingly.
    """
    nROIs, nT = input_list.shape
    rows = []
    start_index = 0

    # Create the shifted rows until the end of the list is reached
    while start_index < nT - row_length:
        # Compute the end index for the current row
        end_index = start_index + row_length
        # Create and append the current row to the rows list
        row = input_list[:, (start_index + end_index) // 2]
        rows.append(row)

        # Update the start index for the next row
        start_index += k

    dataChunked = np.array(rows)
    return dataChunked


def calculateSignalSynchrony(signals, windowSize, stepSize):
    """
    Calculate pairwise synchrony between multiple signals using sliding windows.

    Args:
        signals: List of numpy arrays containing signal data
        windowSize: Number of samples in each window
        stepSize: Number of samples to slide window

    Returns:
        Array of synchrony values for each window
    """
    numWindows = (len(signals[0]) - windowSize) // stepSize + 1
    synchronyValues = []

    for i in range(numWindows):
        start = i * stepSize
        end = start + windowSize
        windowSynchrony = []

        # Calculate pairwise correlations
        for j in range(len(signals)):
            for k in range(j + 1, len(signals)):
                corr = np.corrcoef(signals[j][start:end], signals[k][start:end])[0, 1]
                windowSynchrony.append(corr)

        synchronyValues.append(np.mean(windowSynchrony))

    return np.array(synchronyValues)


def slidingCorrelation(signals, refSignal, windowSize, stepSize):
    numWindows = (len(signals[0]) - windowSize) // stepSize + 1
    correlation = []

    for i in range(numWindows):
        start = i * stepSize
        end = start + windowSize
        windowCorr = []

        # Calculate pairwise correlations
        for j in range(len(signals)):
            corr = abs(np.corrcoef(signals[j][start:end], refSignal[start:end])[0, 1])
            windowCorr.append(corr)

        correlation.append(np.nanmean(windowCorr))

    return np.array(correlation)


def slidingCorrelationDistribution(
    signals, refSignal, windowSize, stepSize, nFolds: int, sampleSize: int
):
    """get distributing for signal-signal correlations
    ---
    Args
    ---
    - signals (# signals x # samples): signals of interest
    - refSignal: reference signal
    - windowSize: correlation window size
    - stepSize: stride between correlation windows
    - nFolds: number of folds to run
    - sampleSize: number of signals to include per fold
    """
    correlations = []

    for i in tqdm(range(nFolds)):
        np.random.seed(i)
        inds = np.arange(len(signals))
        indsSubsample = np.random.choice(
            inds, size=min(len(signals), sampleSize), replace=False
        )
        correlation = slidingCorrelation(
            signals[indsSubsample], refSignal, windowSize, stepSize
        )
        correlations.append(correlation)

    return np.array(correlations)


def slidingWindow(signal, windowSize, stepSize):
    numWindows = (len(signal) - windowSize) // stepSize + 1
    res = []

    for i in range(numWindows):
        start = i * stepSize
        end = start + windowSize

        # Calculate pairwise correlations
        mu = np.nanmean(signal[start:end])
        res.append(mu)

    return np.array(res)


def calcMeanBetween(signal, points):
    # pad the points to start with 0
    points = np.pad(points, (1, 0), mode="constant", constant_values=0)

    # get res
    res = []

    for i in range(len(points[:-1])):
        val = np.nanmean(signal[points[i] : points[i + 1]])
        if np.isnan(val):  # guard against nans
            res.append(res[-1]) if res else res.append(0)
        else:
            res.append(val)

    res = np.pad(res, (0, 1), mode="edge")

    return res
