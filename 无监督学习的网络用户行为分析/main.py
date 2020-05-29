# -*- coding: utf-8 -*-

from ml_algorithms import (
    KMeansAlgorithm,
    RandomForestAlgorithm,
    SVMAlgorithm,
    LogisticRegressionAlgorithm,
    VotingClassifier
)


def real_predict(test_data, rfa, svma, lra, vclf):
    rfa_predeit = rfa.classifier.predict(test_data[0:1])[0]
    svma_predeit = svma.classifier.predict(test_data[0:1])[0]
    lra_predeit = lra.classifier.predict(test_data[0:1])[0]
    vclf_predict = vclf.classifier.predict(test_data[0:1])[0]
    print('RF预测结果：' + str(rfa_predeit))
    print('SVM预测结果：' + str(svma_predeit))
    print('LR预测结果：' + str(lra_predeit))
    print('Voting预测结果：' + str(vclf_predict))
    return rfa_predeit, svma_predeit, lra_predeit, vclf_predict


if __name__ == '__main__':
    kma = KMeansAlgorithm()
    kma.run()

    rfa = RandomForestAlgorithm()
    rfa.run()

    svma = SVMAlgorithm()
    svma.run()

    lra = LogisticRegressionAlgorithm()
    lra.run()

    vclf = VotingClassifier(rfa, svma, lra)
    vclf = vclf.run()

    # 在数据中取出1000条，模仿一组聚类，统计预测结果
    for i in range(1, 1000):
        print('————————————————————')
        real_predict(rfa.test_data[i:i+1], rfa, svma, lra, vclf)
