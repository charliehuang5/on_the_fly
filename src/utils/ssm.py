import sys
import glob
import ssm
import numpy as np
from collections import defaultdict


class SSMCV:
    def __init__(self, data, inputs):
        """
        ---
        params
        ---
        - data: T X nFeatures
        - inputs: T X nFeatures """
        self.data, self.inputs = data, inputs

    def getContiguousInds(self, seed: int, arrLen: int, nSamples: int):
        """
        ---
        params
        ---
        - start: starting index to form chunk
        - nSamples: length of index chunk
        - arrLen: length of full array
        """
        np.random.seed(seed)
        start = np.random.choice(np.arange(arrLen))
        return np.arange(start, start + nSamples) % arrLen

    def doGridSearch(
            self, Ds: tuple, Ls: tuple, maxIter: int = 20, trainFrac: float = 0.8, nFolds:int = 5
    ):
        data = self.data
        inputs = self.inputs
        allElbos = []
        print("doing grid search...")
        for d in Ds:
            dElbo = []
            for l in Ls:
                cvElbos = []
                for c in range(nFolds):
                    # get training and validation inds
                    trainInds = self.getContiguousInds(
                        c, len(data), int(trainFrac * len(data))
                    )
                    validationInds = np.array(list(set(np.arange(len(data))) - set(trainInds)))

                    # init model
                    slds = ssm.SLDS(
                        len(data.T),  # nFeatures X T
                        d,
                        l,
                        M=len(inputs.T),
                        emissions="gaussian_orthog",
                        transitions="recurrent_only",
                        dynamics="gaussian",
                        single_subspace=True,
                    )
                    # Fit the model using Laplace-EM with a structured variational posterior
                    try:
                        _, _ = slds.fit(
                            data[trainInds],
                            method="laplace_em",
                            variational_posterior="structured_meanfield",
                            initialize=False,
                            inputs=inputs[trainInds],
                            num_iters=maxIter,
                        )

                        # Get elbos on held-out data
                        elbos, _ = slds.approximate_posterior(
                            datas=data[validationInds],
                            method="laplace_em",
                            variational_posterior="structured_meanfield",
                            inputs=inputs[validationInds],
                            num_iters=10,
                        )
                    except:
                        continue
                    cvElbos.append(np.max(elbos))
                dElbo.append(np.mean(cvElbos))
            allElbos.append(dElbo)
        print("Done!")
        return allElbos
