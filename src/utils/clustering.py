from src.utils.tracking import resampleRows
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import numpy as np
from scipy.stats import pearsonr
from scipy.linalg import norm
from tqdm import tqdm


def groupByLabel(x, labels):
    uniqueLabels = np.unique(labels)
    sortedLabels = [
        np.argsort(np.sum(x[labels == l], axis=1))[::-1] for l in uniqueLabels
    ]
    sideSideResponsesGrouped = np.vstack(
        [x[labels == l][sortedLabels[i]] for i, l in enumerate(uniqueLabels)]
    )
    return sideSideResponsesGrouped


def cluster_and_plot_functional_data(
    time_series_data,
    percentile=70,
    n_clusters=None,
    figsize=(8, 8),
    featName: str = "",
    doPlot: bool = True,
):
    """
    Cluster time series data and create visualization showing average responses for each cluster.

    Parameters:
    -----------
    time_series_data : array-like
        Array of shape (n_samples, n_timepoints) containing time series data
    percentile : float, optional
        Percentile of heights to use as threshold (default=70).
        Higher percentile = fewer clusters
        Lower percentile = more clusters
        Ignored if n_clusters is specified
    n_clusters : int, optional
        Number of clusters to form. If specified, overrides percentile parameter
    figsize : tuple, default=(8, 12)
        Figure size (width, height)
    """

    # Compute linkage matrix using Ward's method
    linkage_matrix = linkage(time_series_data, method="ward")

    # Get heights from linkage matrix
    heights = linkage_matrix[:, 2]

    # Determine clusters
    if n_clusters is not None:
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust")
        distance_threshold = None  # For dendrogram coloring
    else:
        # Calculate threshold based on percentile
        distance_threshold = np.percentile(heights, percentile)
        cluster_labels = fcluster(
            linkage_matrix, t=distance_threshold, criterion="distance"
        )

    n_clusters_found = len(np.unique(cluster_labels))
    print(f"Number of clusters formed: {n_clusters_found}")

    # Compute average time series for each cluster
    cluster_averages = []
    cluster_stds = []  # For error bands
    cluster_sizes = []  # To track number of series in each cluster
    for i in range(1, n_clusters_found + 1):
        mask = cluster_labels == i
        cluster_data = time_series_data[mask]
        cluster_averages.append(np.mean(cluster_data, axis=0))
        cluster_stds.append(np.std(cluster_data, axis=0))
        cluster_sizes.append(np.sum(mask))

    cluster_averages = np.array(cluster_averages)
    cluster_stds = np.array(cluster_stds)

    if not doPlot:
        return cluster_labels

    # Create color gradient
    colors = plt.cm.cool(np.linspace(0, 1, n_clusters_found))

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = plt.GridSpec(n_clusters_found, 2, width_ratios=[1, 2])

    # Plot dendrogram
    ax_dendrogram = fig.add_subplot(gs[:, 0])
    dendrogram(
        linkage_matrix,
        orientation="left",
        color_threshold=0,
        labels=range(1, len(time_series_data) + 1),
        leaf_rotation=0,
        ax=ax_dendrogram,
        truncate_mode="lastp",
        # p=n_clusters,
        no_labels=True,
    )
    ax_dendrogram.set_xticks([])
    ax_dendrogram.invert_yaxis()
    ax_dendrogram.spines["top"].set_visible(False)
    ax_dendrogram.spines["bottom"].set_visible(False)
    ax_dendrogram.spines["right"].set_visible(False)
    ax_dendrogram.spines["left"].set_visible(False)

    if distance_threshold:
        ax_dendrogram.axvline(
            x=distance_threshold,
            color="r",
            linestyle="--",
            label=f"{percentile}th percentile",
        )
        ax_dendrogram.legend()

    # Plot averaged time series
    # time_points = np.linspace(0, 1, time_series_data.shape[1])
    time_points = np.arange(time_series_data.shape[1])
    time_points = time_points - len(time_points) // 2
    max_val = np.max(np.abs(cluster_averages))
    spacing = 2 * max_val

    ax_series = fig.add_subplot(gs[:, 1])

    # Get unique clusters in dendrogram order
    leaf_order = dendrogram(linkage_matrix, no_plot=True)["leaves"]
    ordered_clusters = [cluster_labels[i] for i in leaf_order]
    unique_clusters_ordered = []
    for cluster in ordered_clusters:
        if cluster not in unique_clusters_ordered:
            unique_clusters_ordered.append(cluster)

    # Plot averaged time series in dendrogram order
    for idx, cluster in enumerate(unique_clusters_ordered):
        y_offset = (n_clusters_found - 1 - idx) * spacing
        mean_curve = cluster_averages[cluster - 1]
        std_curve = cluster_stds[cluster - 1]

        # Plot mean curve
        ax_series.plot(
            time_points,
            mean_curve + y_offset,
            color=colors[cluster - 1],
            linewidth=2,
            label=f"Cluster {cluster} (n={cluster_sizes[cluster-1]})",
        )

        # Add error bands
        ax_series.fill_between(
            time_points,
            mean_curve + y_offset - std_curve,
            mean_curve + y_offset + std_curve,
            color=colors[cluster - 1],
            alpha=0.2,
        )

    # Customize appearance
    ax_series.set_yticks([])
    ax_series.set_xlabel("Time")
    ax_series.spines["left"].set_visible(False)
    ax_series.spines["right"].set_visible(False)
    ax_series.spines["top"].set_visible(False)
    ax_series.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.suptitle(f"Response types for {featName}", y=0.95)
    plt.tight_layout()
    plt.show()


def sampleRandomPoints(nT: int, nPoints: int, seed: int):
    """
    Sample random points within time window

    Parameters:
    -----------
    nT: length of time series
    nPoints: number of points to sample
    seed: random seed

    Returns:
    --------
    pts: randomly sampled points
    """
    np.random.seed(seed)
    arr = np.arange(nT)
    pts = np.random.choice(arr, size=nPoints, replace=False)
    return pts


def getRandomSideSideEpochs(periods: list, nT: int, nEpochs: int, seed: int):
    """
    Get random side-side epochs for null distribution analysis

    Parameters:
    -----------
    periods: cycle periods for side-side stimulus
    nEpochs: number of epochs
    seed: random seed

    Returns:
    --------
    epochPeriods: list of period end points within each epoch
    """
    # randomly sample change points
    cps = sampleRandomPoints(nT=nT, nPoints=nEpochs, seed=seed)

    # get periods within each epoch
    epochPeriods = []
    periodStart = 0

    for i in range(nEpochs):
        epochPeriods.append([p for p in periods if periodStart < p[-1] < cps[i]])
        periodStart = cps[i]

    return epochPeriods


def getMFDistEpochs(events: list, nT: int, nEpochs: int, seed: int, cps=None):
    """
    Get random mfDist epochs for null distribution analysis

    Parameters:
    -----------
    events: time points with large changes in mfDist
    nEpochs: number of epochs
    seed: random seed

    Returns:
    --------
    epochEvents: list of events within each epoch
    """
    # randomly sample change points
    if cps is None:
        cps = sampleRandomPoints(nT=nT, nPoints=nEpochs, seed=seed)

    # get events within each epoch
    epochEvents = []
    eventStart = 0
    winLen = 6  # time pre- and post-change in mfDist

    for i in range(nEpochs):
        epochEvents.append(
            [
                int(e)
                for e in events
                if eventStart < e < cps[i] and e - winLen > 0 and e + winLen < nT
            ]
        )
        eventStart = cps[i]

    return epochEvents


def compareTuningMFDist(activity, epochEvents):
    """
    Get null distribution for changes in tuning

    Parameters:
    -----------

    Returns:
    --------

    """
    winLen = 6
    nEpochs = len(epochEvents)
    corrMat = []
    pMat = []

    # get activity split into epochs
    mfDistEpochs = [
        [activity[:, t - winLen : t + winLen + 1] for t in epoch]
        for epoch in epochEvents
    ]  # nEpochs x nEvents x nROIs x nT

    mfDistEpochMeans = (
        []
    )  # contains trial-averaged activity for each ROI in each behavioral epoch

    # iterate over epochs
    for i in range(nEpochs):
        mfDistEpochMeans.append(
            np.mean(mfDistEpochs[i], axis=0)
        )  # average across trials

    mfDistEpochMeans = np.moveaxis(
        np.array(mfDistEpochMeans), 1, 0
    )  # nT x nEpochs x nROIs
    # import pdb; pdb.set_trace()

    # compute matrix of corr coefs
    for i in range(nEpochs):
        for j in range(nEpochs):
            if j > i:
                corr = np.array([pearsonr(x[i], x[j]) for x in mfDistEpochMeans])
                corrMat.append(corr[:, 0])  # correlation values
                pMat.append(corr[:, 1])  # p values

    return np.array(corrMat), np.array(pMat), np.array(mfDistEpochMeans)


def computeTuningMFDist(activity, events: list, cps, nEpochs, nFolds: int):
    """
    compute null distribution for cosine similiarty

    Parameters:
    -----------


    Returns:
    --------

    """
    winLen = 6
    nT = len(activity.T)
    simMatFolds = []
    seed = 0

    for _ in tqdm(range(nFolds)):
        simMat = []
        # get activity split into epochs
        epochEvents = [[] for _ in range(nEpochs)]

        if cps is None:  # sample random change points
            # continue sampling until we get events in all epochs
            while np.any([len(x) == 0 for x in epochEvents]):
                epochEvents = getMFDistEpochs(events, nT, nEpochs, seed=seed, cps=None)
                seed += 1
        else:  # use actual change points
            epochEvents = getMFDistEpochs(events, nT, nEpochs, seed=seed, cps=cps)

        mfDistEpochs = [
            [activity[:, t - winLen : t + winLen + 1] for t in epoch]
            for epoch in epochEvents
        ]  # nEpochs x nEvents x nROIs x nT

        mfDistEpochMeans = (
            []
        )  # contains trial-averaged activity for each ROI in each behavioral epoch

        # iterate over epochs
        for i in range(nEpochs):
            mfDistEpochMeans.append(
                np.mean(mfDistEpochs[i], axis=0)
            )  # average across trials

        mfDistEpochMeans = np.moveaxis(
            np.array(mfDistEpochMeans), 1, 0
        )  # nT x nEpochs x nROIs
        # import pdb; pdb.set_trace()

        # compute matrix of corr coefs
        for i in range(nEpochs):
            for j in range(nEpochs):
                if j > i:
                    sim = np.array(
                        [
                            np.dot(x[i], x[j]) / (norm(x[i]) * norm(x[j]))
                            for x in mfDistEpochMeans
                        ]
                    )
                    simMat.append(sim)
        simMatFolds.append(simMat)

    simMatFolds = np.transpose(
        np.array(simMatFolds), (1, 2, 0)
    )  # n epoch comparisons X nROIs X nFolds

    return simMatFolds
