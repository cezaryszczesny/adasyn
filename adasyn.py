from collections import Counter

import numpy as np
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.neighbors import NearestNeighbors


# Authors: Dominik Badora, Cezary Szczesny
class Adasyn(BaseOverSampler):

    def __init__(self):
        pass

    def _fit_resample(self, X, y, n_neighbors=5, ratio=0.9):
        examplesInEachClass = Counter(y)
        maxNumberOfSample = max(examplesInEachClass.values())
        syntheticExampleList = []

        for classIdx in examplesInEachClass.keys():
            sampleIndexes = np.where(y == classIdx)[0]
            currentClassSamples = X[sampleIndexes]
            syntheticSamplesToGenerateInClass = int(ratio * (maxNumberOfSample - examplesInEachClass[classIdx]))

            if syntheticSamplesToGenerateInClass == 0:
                continue

            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(currentClassSamples)
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
