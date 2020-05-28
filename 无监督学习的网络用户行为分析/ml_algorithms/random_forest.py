# -*- coding: utf-8 -*-
import matplotlib
import matplotlib.pyplot as plt
from ml_algorithms.ml_algorithm_interface import AlgorithmInterface
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, auc
from sklearn.model_selection import RandomizedSearchCV
from collections import OrderedDict

matplotlib.use('TkAgg')

class RandomForestAlgorithm(AlgorithmInterface):
    def __init__(self):
        self.columns = OrderedDict()
        super(RandomForestAlgorithm, self).__init__()

    def feature_engineering(self):
        self.convert_symbolic_feature_into_continuous()

    def train_phase(self):
        random_forest = RandomForestClassifier()

        # 森林中树个数
        # n_estimators = [100, 500, 900, 1100, 1500]
        n_estimators = [100]

        # 每个树的最大深度
        # max_depth = [10, 15, 20, 25]
        max_depth = [10, 20]

        # 调超参
        hyper_parameter_grid = {'n_estimators': n_estimators,
                                'max_depth': max_depth}

        # 设置4折交叉验证随机搜索
        self.classifier = RandomizedSearchCV(estimator=random_forest,
                                             param_distributions=hyper_parameter_grid,
                                             cv=4, n_iter=1,
                                             scoring='roc_auc',
                                             n_jobs=-1, verbose=2,
                                             return_train_score=True,
                                             random_state=42)

        # 开始训练
        self.classifier.fit(self.train_data, self.train_label)
        print("训练结束")

    def test_phase(self):
        self.y_predict = self.classifier.predict(self.test_data)

        self.accuracy_score = accuracy_score(self.test_label, self.y_predict)
        print("准确度: %f" % self.accuracy_score)

        self.precision_score = precision_score(self.test_label, self.y_predict, average="macro")
        print("精确度: %f" % precision_score(self.test_label, self.y_predict, average="macro"))

        self.recall_score = recall_score(self.test_label, self.y_predict, average="macro")
        print("召回率: %f" % recall_score(self.test_label, self.y_predict, average="macro"))


    def show(self):
        fpr, tpr, thresholds = metrics.roc_curve(self.y_predict, self.test_label)
        plt.plot(fpr, tpr, marker='o')
        plt.show()
        auc_score = auc(fpr, tpr)
        print("AUC: %f" % auc_score)
