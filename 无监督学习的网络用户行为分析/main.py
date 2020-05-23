# -*- coding: utf-8 -*-

from ml_algorithms import (
    KMeansAlgorithm,
    RandomForestAlgorithm,
    SVMAlgorithm,
    LogisticRegressionAlgorithm
)

if __name__ == '__main__':
    kma = KMeansAlgorithm()
    kma.run()

    rfa = RandomForestAlgorithm()
    rfa.run()

    svma = SVMAlgorithm()
    svma.run()

    lra = LogisticRegressionAlgorithm()
    lra.run()
