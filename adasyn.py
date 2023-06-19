from collections import Counter

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors


# Authors: Dominik Badora, Cezary Szczesny
class Adasyn(BaseOverSampler):

    def __init__(self, sampling_strategy="auto", n_neighbors=5, ratio=1.0):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_neighbors = n_neighbors
        self.ratio = ratio

    def _fit_resample(self, X, y):
        examplesInEachClass = Counter(y)
        maxNumberOfSample = max(examplesInEachClass.values())
        syntheticExampleList = []

        for classIdx in examplesInEachClass.keys():
            sampleIndexes = np.where(y == classIdx)[0]
            currentClassSamples = X[sampleIndexes]
            syntheticSamplesToGenerateInClass = int(self.ratio * (maxNumberOfSample - examplesInEachClass[classIdx]))

            if syntheticSamplesToGenerateInClass == 0:
                continue

            knn = NearestNeighbors(n_neighbors=self.n_neighbors).fit(currentClassSamples)
            knnOfExample = knn.kneighbors(currentClassSamples, return_distance=False)

            for classSample in range(syntheticSamplesToGenerateInClass):

                randomExample = np.random.randint(len(sampleIndexes))
                exampleRandNNeighbor = np.random.choice(knnOfExample[randomExample])
                diffrenceBetweenExampleAndNeighbor = currentClassSamples[exampleRandNNeighbor] - currentClassSamples[
                    randomExample]
                syntheticExample = currentClassSamples[
                                       randomExample] + np.random.rand() * diffrenceBetweenExampleAndNeighbor

                if examplesInEachClass[classIdx] < maxNumberOfSample:
                    syntheticExampleList.append(syntheticExample)
                    examplesInEachClass[classIdx] += 1
                    y = np.concatenate((y, np.full(1, classIdx)))
                    X = np.concatenate((X, np.array([syntheticExample])))
        return X, y
