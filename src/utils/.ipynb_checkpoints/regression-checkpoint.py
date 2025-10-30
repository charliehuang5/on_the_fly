import numpy as np
from sklearn.linear_model import RidgeCV
from tqdm import tqdm
from scipy.stats import zscore
from src.utils.tracking import *
from src.utils.fictrac import mergeBouts
from collections import defaultdict
import statsmodels.api as sm
from scipy.linalg import block_diag, svd


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


def doRidgeCV(
    X,
    Y,
    nFolds: int = 100,
    trainFrac: float = 0.8,
    alphas: tuple = (0.1, 1, 10, 100, 1000, 10000, 100000),
):
    """fit ridge regression model and evaluate performance on held-out data
    ---
    Inputs
    ---
    - X: nSamples x nFeatures
    - Y: nSamples"""

    # train + evaluate on multiple folds
    scores = []

    for i in tqdm(range(nFolds)):
        # partition data
        np.random.seed(i)
        inds = np.arange(len(X))
        np.random.shuffle(inds)  # shuffle the inds
        indsTrain = inds[: int(len(inds) * trainFrac)]
        indsValidation = inds[int(len(inds) * trainFrac) :]
        XTrain = X[indsTrain]
        XValidation = X[indsValidation]
        YTrain = Y[indsTrain]
        YValidation = Y[indsValidation]

        # fit model
        clf = RidgeCV(alphas=alphas)
        clf.fit(XTrain, YTrain)

        # evaluate model
        score = clf.score(XValidation, YValidation)
        scores.append(score)

    return scores


def makeRegressionData(
    flyvrData, cnnData, fictracTimestamps: list, featureList: list, fullLen=False
):
    print("Creating regression data...")
    speedDict, vidData, favs = (
        flyvrData["speedDict"],
        flyvrData["vidData"],
        flyvrData["favs"],
    )
    if cnnData is not None:
        wingBinaryL, wingBinaryR, tapBinary = (
            cnnData["wingBinaryL"],
            cnnData["wingBinaryR"],
            cnnData["tapBinary"],
        )
    else:
        wingBinaryL = wingBinaryR = tapBinary = np.zeros_like(
            speedDict["rotational_speed"]
        )

    # get speed
    mRS, mFS, mLS = (
        speedDict["rotational_speed"],
        speedDict["forward_speed"],
        speedDict["lateral_speed"],
    )
    mFS = abs(mFS)
    mLS = abs(mLS)
    mFV = mFS.copy()

    # desired fictrac time
    desiredTimeFictrac = np.arange(
        len(speedDict["rotational_speed"])
    )  # fictrac samples for each video frame

    # load actuator data
    stim = vidData["video"]["stimulus"]["actuator"]
    mfDist = stim[:, 3]
    mfDist = max(mfDist) - mfDist  # convert to actual distance
    sideSide = stim[:, 4]
    sideSide -= np.median(sideSide)  # remove offset
    femVel = np.diff([0] + list(mfDist))

    # downsample to fictrac timebase
    mfDistFic = np.interp(desiredTimeFictrac, favs, mfDist)
    sideSideFic = np.interp(desiredTimeFictrac, favs, sideSide)
    femSpeedFic = np.interp(desiredTimeFictrac, favs, femVel)

    # also pull out left and right side-side
    sideSideL = np.where(sideSideFic > 0, 0, abs(sideSideFic))  # left signal
    sideSideR = np.where(sideSideFic < 0, 0, abs(sideSideFic))  # right signal

    # get left and right taps
    minLen = min(len(tapBinary), len(sideSideFic))
    tapL = tapBinary[:minLen] * np.where(sideSideFic > 0, 0, 1)[:minLen]
    tapR = tapBinary[:minLen] * np.where(sideSideFic < 0, 0, 1)[:minLen]

    # pull out left and right side-side at least 2mm from midline
    sideSideLX = np.where(sideSideFic > 2, 1, 0)  # left signal
    sideSideRX = np.where(sideSideFic < -2, 1, 0)  # right signal

    # binned side-side motion
    sideSide4P = np.where((3 < sideSideFic) & (sideSideFic < 4), 1, 0)
    sideSide3P = np.where((2 < sideSideFic) & (sideSideFic < 3), 1, 0)
    sideSide2P = np.where((1 < sideSideFic) & (sideSideFic < 2), 1, 0)
    sideSide1P = np.where((0 < sideSideFic) & (sideSideFic < 1), 1, 0)
    sideSide4N = np.where((-3 > sideSideFic) & (sideSideFic > -4), 1, 0)
    sideSide3N = np.where((-2 > sideSideFic) & (sideSideFic > -3), 1, 0)
    sideSide2N = np.where((-1 > sideSideFic) & (sideSideFic > -2), 1, 0)
    sideSide1N = np.where((0 > sideSideFic) & (sideSideFic > -1), 1, 0)

    # pull out central region of visual field
    sideSideCenter = np.where(abs(sideSideFic) < 2, 1, 0)

    # pull out left and right turns
    mRSL = np.where(mRS > 0, 0, abs(mRS))  # left turns
    mRSR = np.where(mRS < 0, 0, abs(mRS))  # right turns

    # pull out large left and right turns
    mRSLX = np.where(zscore(mRS) >= 2, 1, 0)  # left turns
    mRSRX = np.where(zscore(mRS) <= -2, 1, 0)  # right turns

    # get left/right turn bouts
    leftTurnLabels, _ = label(mRSLX)
    rightTurnLabels, _ = label(mRSRX)
    _, leftTurnBouts = mergeBouts(
        leftTurnLabels, fps=90, minBoutDist=1000, minBoutLen=500
    )
    _, rightTurnBouts = mergeBouts(
        rightTurnLabels, fps=90, minBoutDist=1000, minBoutLen=500
    )
    turnBoutsAll = leftTurnBouts + rightTurnBouts
    turnBoutsAll = np.where(turnBoutsAll > 0, 1, 0)

    # downsample fFV using averaging
    fFSNeg = np.where(femSpeedFic < 0, 1, 0)
    fFSPos = np.where(femSpeedFic > 0, 1, 0)

    # get tracking data
    ti, changePoints = getTrackingIndex(flyvrData, returnEpochs=True)

    # get periods where tracking mean is greater than median
    trackingThresh = 0.4
    trackHigh = np.where(ti > trackingThresh, 1, 0)
    trackLow = np.where(ti < trackingThresh, 1, 0)

    # get rotational speed during high and low tracking periods
    mRSRZHighT = zscore(mRSR * trackHigh, nan_policy="omit")
    mRSLZHighT = zscore(mRSL * trackHigh, nan_policy="omit")
    mRSRZLowT = zscore(mRSR * trackLow, nan_policy="omit")
    mRSLZLowT = zscore(mRSL * trackLow, nan_policy="omit")
    mRSLZHighT -= np.median(mRSLZHighT)
    mRSRZHighT -= np.median(mRSRZHighT)
    mRSLZLowT -= np.median(mRSLZLowT)
    mRSRZLowT -= np.median(mRSRZLowT)

    # get binarized tracking index
    tiBinary = np.where(ti >= trackingThresh, 1, 0)

    # collect data
    regressionData = {
        "mfDistZ": zscore(mfDistFic, nan_policy="omit"),
        "sideSideZ": zscore(sideSideFic, nan_policy="omit"),
        "sideSideLZ": zscore(sideSideL, nan_policy="omit"),
        "sideSideRZ": zscore(sideSideR, nan_policy="omit"),
        "sideSideLXZ": zscore(sideSideLX, nan_policy="omit"),
        "sideSideRXZ": zscore(sideSideRX, nan_policy="omit"),
        "sideSideCenterZ": zscore(sideSideCenter, nan_policy="omit"),
        "fFSNegZ": zscore(fFSNeg, nan_policy="omit"),
        "fFSPosZ": zscore(fFSPos, nan_policy="omit"),
        "mRSZ": zscore(mRS, nan_policy="omit"),
        "mRSLZ": zscore(mRSL, nan_policy="omit"),
        "mRSRZ": zscore(mRSR, nan_policy="omit"),
        "mRSLX": mRSLX,  # extreme left, binary
        "mRSRX": mRSRX,  # extreme right, binary
        "mRSLZHighT": mRSLZHighT,  # left turns during high tracking
        "mRSRZHighT": mRSRZHighT,  # right turns during high tracking
        "mRSLZLowT": mRSLZLowT,  # left turns during low tracking
        "mRSRZLowT": mRSRZLowT,  # right turns during low tracking
        "mFSZ": zscore(mFS, nan_policy="omit"),
        "mFVZ": zscore(mFV, nan_policy="omit"),
        "mLSZ": zscore(mLS, nan_policy="omit"),
        "tapZ": zscore(tapBinary, nan_policy="omit"),
        "wingLZ": zscore(wingBinaryL, nan_policy="omit"),
        "wingRZ": zscore(wingBinaryR, nan_policy="omit"),
        "trackingIndexZ": zscore(ti, nan_policy="omit"),
        "mfDist": mfDistFic,
        "sideSideL": sideSideL,
        "sideSideR": sideSideR,
        "sideSideLX": sideSideLX,
        "sideSideRX": sideSideRX,
        "sideSide": sideSideFic,
        "sideSideCenter": sideSideCenter,
        "sideSide4P": sideSide4P,
        "sideSide3P": sideSide3P,
        "sideSide2P": sideSide2P,
        "sideSide1P": sideSide1P,
        "sideSide4N": sideSide4N,
        "sideSide3N": sideSide3N,
        "sideSide2N": sideSide2N,
        "sideSide1N": sideSide1N,
        "mRS": mRS,
        "fFSNegZ": fFSNeg,
        "fFSPosZ": fFSPos,
        "mRSL": mRSL,
        "mRSR": mRSR,
        "turnBoutL": leftTurnBouts,
        "turnBoutR": rightTurnBouts,
        "turnBoutsAll": turnBoutsAll,
        "mFS": mFS,
        "mFV": mFV,
        "mLS": mLS,
        "tap": tapBinary,
        "tapL": tapL,
        "tapR": tapR,
        "wingL": wingBinaryL,
        "wingR": wingBinaryR,
        "trackingIndex": ti,
        "trackingIndexBinary": tiBinary,
    }
    print("Done!")
    if not fullLen:
        return {
            k: v[fictracTimestamps]
            for k, v in regressionData.items()
            if k in featureList and k in regressionData
        }
    else:
        return {
            k: v
            for k, v in regressionData.items()
            if k in featureList and k in regressionData
        }


def doBHCorrection(pVals: list, Q=0.05):
    """
    Benjamini-Hochberg (BH) multiple comparisons correction

    Parameters:
    -----------
    pVals: p values
    Q: false discovery rate

    Returns:
    --------
    whichSig: indices for significant p values
    whichInsig: indices for insignificant p values
    """
    nTests = len(pVals)  # number of statistical tests
    pValsSorted = pVals.copy()  # make copy of pvals
    indsSorted = np.argsort(pValsSorted)  # ordering of p values
    pValsSorted = np.sort(pValsSorted)  # p values sorted from smallest to largest
    values = [Q * (r + 1) / nTests for r in range(nTests)]  # BH values
    whereCritical = np.argwhere(
        pValsSorted <= values
    )  # get the largest p value that is less than its critical value
    if len(whereCritical) == 0:
        return [], []
    else:
        whereCritical = whereCritical[-1][0]
    whichSig = indsSorted[: whereCritical + 1]  # inds up to an including critical index
    whichInsig = indsSorted[whereCritical + 1 :]  # inds after critical index

    return whichSig, whichInsig


def runRegression(regressionData, allNeuralData, cutoff=0.05, correction="BH"):
    assert correction in [
        "bonferroni",
        "BH",
    ], "invalid multiple comparisons correction method"
    XSimple = np.vstack([feat for feat in regressionData.values()])
    XCSimple = np.concatenate((np.ones(len(XSimple.T))[None, :], XSimple)).T
    allPVals = []
    modelParams = defaultdict(list)

    # Fit a separate model for each response variable.
    # exog: A nobs x k array where nobs is the number of observations and k is the number of regressors. An intercept is not included by default and should be added by the user.
    print("Fitting regression models...")
    featureNames = list(regressionData.keys())

    for i in tqdm(range(len(allNeuralData))):
        model = sm.OLS(allNeuralData[i, : len(XCSimple)], XCSimple).fit()
        pvals = model.pvalues
        params = model.params

        # collect p values
        allPVals.append(pvals)

        # collect params (exclude the intercept)
        for i, p in enumerate(params[1:]):
            modelParams[featureNames[i]].append(p)

    allPVals = np.array(allPVals)
    sigMod = {}

    for i, featName in enumerate(regressionData.keys()):
        if correction == "bonferroni":
            sigMod[featName] = np.squeeze(
                np.where(
                    allPVals[:, i + 1]
                    < cutoff / (len(allNeuralData) * len(regressionData))
                )
            )
        elif correction == "BH":
            pvals = allPVals[:, 1 + i]  # exclude intercept
            whichSig, _ = doBHCorrection(pvals)
            sigMod[featName] = whichSig

    # collect regression results
    regressionResults = {"sigMod": sigMod, "modelParams": modelParams}
    print("Done!")
    return regressionResults


def correlationAnalysis(regressionData, featName, allNeuralData, mcCorrection=True):
    featData = regressionData[featName]
    corrs = []
    pVals = []

    for i in tqdm(range(len(allNeuralData))):
        corr, pval = pearsonr(allNeuralData[i], featData)
        corrs.append(corr)
        pVals.append(pval)

    corrs, pVals = np.array(corrs), np.array(pVals)

    # do multiple comparisons correction if specified
    if mcCorrection:
        pVals = pVals * len(allNeuralData)

    return corrs, pVals
