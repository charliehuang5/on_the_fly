import glob
import numpy as np
from scipy.signal import find_peaks


def loadNeuralData(path, channel=1):
    assert channel in [1, 2], f"invalid  channel: {channel}"
    fname = glob.glob(f"{path}/**/channel_{channel}/**/cleaned*.mmap", recursive=True)[
        0
    ]
    split = np.array(fname.split("__")[1].split("_"))
    X, Y, Z = tuple(split[[1, 3, 5]].astype(int))
    T = int(split[-2])
    images = np.memmap(
        fname, shape=(Z, T, X, Y), mode="r+", dtype=np.float32
    )  # Z, T, X, Y
    images = np.transpose(images, (1, 0, -2, -1))  # T, Z, X, Y
    return images


def getSliceAvg(brainStack, zIndex):
    brainSlice = brainStack[:, zIndex, :, :]
    mu = np.mean(brainSlice, axis=0)

    return mu


def getMirrorPeaks(daq, distance=100):
    """Get peaks in galvo mirror."""

    # Load mirror data
    mirror = daq["daq"]["input"]["samples"][:, 1]

    # Find mirror peaks
    peaks = find_peaks(mirror, distance=100, prominence=3)
    peaks = np.array([p for p in peaks[0] if mirror[p] > 0])

    # plt.plot(mirror);plt.scatter(peaks,mirror[peaks],c='r')
    # plt.show()
    # import pdb; pdb.set_trace()

    return peaks


def getVAAS(daq, nSlices: int):
    """get volume at audio sample"""

    # Timestamps for stim and fictrac
    sync = daq["daq"]["chunk_synchronization_info"]
    nDaqSamples = np.nanmax(sync[:, 1])  # number of daq samples

    # Get mirror signal
    daqPoints = getMirrorPeaks(daq)

    # Make dictionary
    start = 0  # starting sample
    vaas = np.array([-1 for ii in range(nDaqSamples)])  # initialize map
    volStarts = [
        daqPoints[0]
    ]  # audio samples corresponding to volume starts. init w/ first mirror sample
    frame = 0

    # New volume after mirror sweeps across xy extent of sample "nSlices" times
    for ii, sample in enumerate(daqPoints):
        isMultiple = (ii + 1) % nSlices == 0
        if isMultiple:
            frame += 1
            volStarts.append(sample)
        vaas[start:sample][:] = frame
        start = sample

    vaas = vaas.astype(int)
    return vaas, volStarts

