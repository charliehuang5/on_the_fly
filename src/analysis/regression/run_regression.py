# % imports
import sys 
sys.path.insert(0, "/Users/mjaragon/Documents/github/courtship_dynamics/")
from src.utils.fictrac import *
from src.utils.data_loading import *
from src.utils.neural_data import *
from src.utils.movies import *
from src.utils.bootstrap import *
from src.utils.regression import *
from src.utils.tracking import *
from src.utils.sliding_windows import *
import argparse
import pickle
import os

# % parse args
parser = argparse.ArgumentParser()
parser.add_argument("--dir", type=str, required=True, help="data directory")
args = parser.parse_args()


# % main
def main():
    # load data
    allNeuralData, idxToROI, roiToIdx = loadSupervoxelData(args.dir, getRaw=False)
    flyvrData = loadFlyVRData(args.dir)
    cnnData = loadCNNPredictions(args.dir, tapThresh=0.6, wingThresh=0.6)
    fsav = getFictracSampsAtVolume(flyvrData)  # fictrac samples for each imaging volume

    # prepare regression data
    regressionData = makeRegressionData(
        flyvrData=flyvrData,
        cnnData=cnnData,
        fictracTimestamps=fsav,
        featureList=[
            "mfDistZ",
            "sideSide4P",
            "sideSide3P",
            "sideSide2P",
            "sideSide1P",
            "sideSide4N",
            "sideSide3N",
            "sideSide2N",
            "sideSide1N",
            "sideSideCenter",
            "turnBoutL",
            "turnBoutR",
            "mFSZ",
            "tapL",
            "tapR",
            "wingL",
            "wingR",
            "trackingIndex",
        ],
    )

    # run regression
    regressionResults = runRegression(regressionData, allNeuralData, cutoff=0.05)
    regressionResults["idxToROI"] = idxToROI
    regressionResults["roiToIdx"] = roiToIdx

    # save results
    saveDir = os.path.dirname(
        glob.glob(args.dir + "/**/supervoxel*.pkl", recursive=True)[0]
    )

    with open(saveDir + "/regression_results.pkl", "wb") as outFile:
        pickle.dump(regressionResults, outFile)

    print(f"Saved regression results to {saveDir}")


if __name__ == "__main__":
    main()
