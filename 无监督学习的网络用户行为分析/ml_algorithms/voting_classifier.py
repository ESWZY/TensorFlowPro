# -*- coding: utf-8 -*-
from sklearn.ensemble import VotingClassifier
from ml_algorithms.ml_algorithm_interface import AlgorithmInterface


class VotingClassifier3(AlgorithmInterface):
    def __init__(self, rfa, svma, lra):
        super(VotingClassifier3, self).__init__()
        self.accuracy_score = 0
        self.classifier = VotingClassifier(estimators=[
            ('rfa', rfa.classifier),
            ('svma', svma.classifier),
            ('lra', lra.classifier)
        ])

    def feature_engineering(self):
        self.convert_symbolic_feature_into_continuous()

    def train_phase(self):
        self.classifier.fit(self.test_data, self.test_label)

    def test_phase(self):
        self.accuracy_score = self.classifier.score(self.test_data, self.test_label)
        print("准确度: %f" % self.accuracy_score)
