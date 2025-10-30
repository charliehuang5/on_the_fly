import numpy as np 

def computeBootstrappedCIs(x, ci=0.95, nRepeats=1000, scale=False):
    """Computes bootstrapped confidence interval of mean.
    ------
    Inputs
    ------
    - x (array): input data (nTrials x nSamples)
    - ci (float): confidence interval
    - nRepeats (int): number of bootstrapped samples
    -------
    Outputs
    -------
    - mu (float): mean of data
    - ub (float): upper bound of ci
    - lb (float): lower bound of ci
    """
    mu = np.nanmean(x, axis=0)
    sampleStats = []
    sampleIDs = np.arange(len(x))  # row IDs for sampling

    # Compute bootrapped sample statistics
    for ii in range(nRepeats):

        # Create bootstrapped sample
        ids = np.random.choice(sampleIDs, size=len(x), replace=True)
        sample = x[ids]

        # Compute sample statistic
        sMu = np.nanmean(sample, axis=0)
        sampleStats.append(sMu)

    # Get upper and lower limits
    sampleStats = np.array(sampleStats)
    ub = np.array([np.percentile(s, 100 * (1 + ci) / 2) for s in sampleStats.T])
    lb = np.array([np.percentile(s, 100 * (1 - ci) / 2) for s in sampleStats.T])

    # scale if specified
    if scale:
        mu0 = mu.copy()
        mu = (mu - np.min(mu)) / (np.max(mu) - np.min(mu))
        ub = (ub - np.min(mu0)) / (np.max(mu0) - np.min(mu0))
        lb = (lb - np.min(mu0)) / (np.max(mu0) - np.min(mu0))

    return mu, ub, lb
