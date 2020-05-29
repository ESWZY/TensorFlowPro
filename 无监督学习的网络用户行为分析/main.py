# -*- coding: utf-8 -*-
from sklearn.ensemble import VotingClassifier

from ml_algorithms import (
    KMeansAlgorithm,
    RandomForestAlgorithm,
    SVMAlgorithm,
    LogisticRegressionAlgorithm
)


def real_predict(test_data, rfa, svma, lra, vclf):
    rfa_predeit = rfa.classifier.predict(test_data[0:1])[0]
    svma_predeit = svma.classifier.predict(test_data[0:1])[0]
    lra_predeit = lra.classifier.predict(test_data[0:1])[0]
    vclf_predict = vclf.predict(test_data[0:1])[0]
    print('RF预测结果：' + str(rfa_predeit))
    print('SVM预测结果：' + str(svma_predeit))
    print('LR预测结果：' + str(lra_predeit))
    print('Voting预测结果：' + str(vclf_predict))


if __name__ == '__main__':
    kma = KMeansAlgorithm()
    kma.run()

    rfa = RandomForestAlgorithm()
    rfa.run()

    svma = SVMAlgorithm()
    svma.run()

    lra = LogisticRegressionAlgorithm()
    lra.run()

    test_data = rfa.test_data
    test_label = rfa.test_label

    vclf = VotingClassifier(estimators=[('rfa', rfa.classifier), ('svma', svma.classifier), ('lra', lra.classifier)])
    vclf = vclf.fit(rfa.test_data, rfa.test_label)
    accuracy_score = vclf.score(test_data, test_label)
    print("准确度: %f" % accuracy_score)

    for i in range(1, 1000):
        print('————————————————————')
        real_predict(test_data[i:i+1], rfa, svma, lra, vclf)
