import numpy as np
from scipy.signal import savgol_filter
from scipy.optimize import minimize
from sklearn.metrics import r2_score
from collections import defaultdict
from scipy.stats import zscore
from tqdm import tqdm


class LinearModel:
    def __init__(self, data, mode: str = "standard"):
        assert mode in ["standard", "interaction", "hierarchical", "modulation"]

        # set hyperparameter bounds
        self.data = data
        self.trackingThresh = 0.1
        self.speedThresh = 2
        self.fitParams = None
        self.mode = mode

    def computeLL(self, y, yHat):
        """
        Calculate log-likelihood for a linear model with normally distributed errors

        Parameters:
        y: array-like of actual values
        yHat: array-like of predicted values

        Returns:
        float: Log-likelihood value
        """
        n = len(y)
        residuals = y - yHat

        # Estimate variance of residuals
        variance = np.var(residuals)

        # Log-likelihood calculation
        ll = (
            -0.5 * variance**-1 * np.sum(residuals**2)
            - 0.5 * n * np.log(variance)
            - 0.5 * n * np.log(2 * np.pi)
        )

        return ll

    def run(self, params: tuple, data, tol=1e-3, evaluate=False, **kwargs):
        """run the model with given hyperparameters"""
        # unpack params
        (
            intercept,
            betaVisLow,
            betaVisHigh,
            betaSpeedLow,
            betaSpeedHigh,
            betaInteractionLow,
            betaInteractionHigh,
        ) = params
        visParams = [betaVisLow, betaVisHigh]
        speedParams = [betaSpeedLow, betaSpeedHigh]
        interactionParams = [betaInteractionLow, betaInteractionHigh]

        # unpack data
        speed = data["speed"]  # male's speed
        tracking = data["tracking"]  # tracking index
        trackingOrig = data["trackingOrig"]  # tracking index
        vis = data["vis"]  # visual cue

        # parameter timeseries
        visSeries = np.array([visParams[s] for s in tracking])
        speedSeries = np.array([speedParams[s] for s in tracking])
        interactionSeries = np.array([interactionParams[s] for s in tracking])

        # compute estimated activity
        if self.mode == "standard":
            Y_hat = intercept + betaVisLow * vis + betaSpeedLow * speed
        elif self.mode == "interaction":
            Y_hat = (
                intercept
                + betaVisLow * vis
                + betaSpeedLow * speed
                + betaInteractionLow * (vis * speed)
            )
        elif self.mode == "hierarchical":
            Y_hat = (
                intercept
                + visSeries * vis
                + speedSeries * speed
                + interactionSeries * (vis * speed)
            )
        elif self.mode == "modulation":
            Y_hat = (
                intercept
                + betaVisLow * vis
                + betaSpeedLow * speed
                + betaInteractionLow * trackingOrig * (vis * speed)
            )

        # compute r^2
        ll = self.computeLL(data["activity"], Y_hat)

        if evaluate:
            r2 = r2_score(data["activity"], Y_hat)
            return r2, np.exp(
                ll / len(Y_hat)
            )  # coef. of determination and likelihood/sample

        return -ll

    def optimize(self, data, **kwargs):
        """optimize the model parameters"""

        # initialize parameters
        np.random.seed(0)
        x0 = np.random.normal(size=7)

        # define function to optimize
        optimFunc = lambda x0: self.run(x0, data, **kwargs)

        # optimize
        options = {}
        res = minimize(
            optimFunc,
            x0,
            method="BFGS",
            options=options,
            # optimFunc, x0, bounds=[*self.bounds], method="L-BFGS-B", options=options
        )

        # Run model with best-fit parameters
        params = res.x
        self.fitParams = params
        return params

    def score(self, data):
        if self.fitParams is None:
            raise ValueError("parameters not fit yet")
        r2, likelihoodPerSamp = self.run(self.fitParams, data=data, evaluate=True)

        return r2, likelihoodPerSamp

    def prepareCVData(
        self,
        heldoutName: str,
        runThresh: float = 2.0,
        trackThresh: float = 0.1,
    ):
        cvData = defaultdict(dict)
        allSpeed = []
        allTracking = []
        allTrackingOrig = []
        allVis = []
        allNeural = []

        # get training data
        for expName in self.data:
            if expName == heldoutName:
                continue
            regressionData = self.data[expName]["regressionData"]

            # get tracking index
            trackingIndex = regressionData["trackingIndex"]
            trackingIndexOrig = trackingIndex.copy()  #
            trackingIndex = np.where(
                trackingIndex > np.percentile(trackingIndex, 50), 1, 0
            )
            allTracking.extend(trackingIndex)
            allTrackingOrig.extend(trackingIndexOrig)

            # get running speed
            mSpeed = savgol_filter(
                np.sqrt(regressionData["mRS"] ** 2 + regressionData["mFS"] ** 2),
                21,
                polyorder=2,
            )
            allSpeed.extend(mSpeed)

            # get visual cue
            vis = self.data[expName]["regressionData"]["mfDistZ"]
            allVis.extend(vis)

            # get neural data
            neuralData = self.data[expName]["pca"]
            allNeural.extend(neuralData)

        # binarize data
        allSpeedOrig = np.copy(allSpeed)
        allSpeed = np.where(np.array(allSpeed) > runThresh, 1, 0)

        minLen = min([len(allSpeed), len(allTracking), len(allVis), len(allNeural)])
        cvData["train"]["speed"] = allSpeed[:minLen]
        cvData["train"]["speedOrig"] = allSpeedOrig[:minLen]
        cvData["train"]["tracking"] = np.array(allTracking)[:minLen]
        cvData["train"]["trackingOrig"] = np.array(allTrackingOrig)[:minLen]
        cvData["train"]["vis"] = zscore(np.array(allVis)[:minLen])
        cvData["train"]["activity"] = zscore(np.array(allNeural)[:minLen])

        # get validation data
        if heldoutName is None:
            return cvData

        # get tracking index
        regressionData = self.data[heldoutName]["regressionData"]
        trackingIndex = regressionData["trackingIndex"]
        trackingIndexOrig = trackingIndex.copy()

        # get running speed
        mSpeed = savgol_filter(
            np.sqrt(regressionData["mRS"] ** 2 + regressionData["mFS"] ** 2),
            21,
            polyorder=2,
        )

        # get visual stim
        vis = self.data[heldoutName]["regressionData"]["mfDistZ"]

        # get neural data
        neuralData = self.data[heldoutName]["pca"]

        # binarize data
        mSpeed = np.where(np.array(mSpeed) > runThresh, 1, 0)
        trackingIndex = np.where(trackingIndex > np.percentile(trackingIndex, 50), 1, 0)

        # update cv dict
        minLen = min([len(mSpeed), len(trackingIndex), len(vis), len(neuralData)])
        cvData["validation"]["speed"] = mSpeed[:minLen]
        cvData["validation"]["tracking"] = np.array(trackingIndex)[:minLen]
        cvData["validation"]["trackingOrig"] = np.array(trackingIndexOrig)[:minLen]
        cvData["validation"]["vis"] = zscore(np.array(vis)[:minLen])
        cvData["validation"]["activity"] = zscore(np.array(neuralData)[:minLen])

        return cvData

    def runCV(self):
        res = {"r2": [], "llPerSamp": [], "params": []}
        print("Running cross-validation...")

        for expName in tqdm(self.data.keys()):
            cvData = self.prepareCVData(heldoutName=expName)  # prepare data
            self.optimize(data=cvData["train"])  # run optimization
            r2, llPerSamp = self.score(
                data=cvData["validation"]
            )  # evaluate on held-out data
            res["r2"].append(r2)
            res["llPerSamp"].append(llPerSamp)
            res["params"].append(self.fitParams)
            self.fitParams = None  # reset the best-fit params

        print("Done!")
        return res
