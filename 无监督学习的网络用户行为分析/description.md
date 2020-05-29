# 实验

该部分是本项目实验部分的介绍。该实验使用Python作为实验程序的代码语言，所需依赖包及其版本号见根目录[requirements.txt](/requirements.txt)文件。

## 实验设计

在上机代码，基于K-Means的无监督学习的网络用户行为分析的架构上，我们能够用户行为数据条目的聚类。但是，仅凭这些聚类，我们很难直观看出哪些聚类中的行为是攻击行为，哪些不是。

所以，在进一步的实验中，我们添加了有监督学习的分类模型，以达到对用户行为聚类的检测与识别。

## 项目文件

### 文件夹 KDD99

该文件夹下存放了上机实验的KDD99数据集[data_10000.csv](/无监督学习的网络用户行为分析/KDD99/data_10000.csv)，也存放了来自[KDD Cup 99数据集](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html) 的官方原始数据集。

### 文件夹 ml_algorithms

该文件夹下存放了上机实验的核心实验代码。

#### 文件 __init__.py

用于项目代码包的导入。

#### 文件 ml_algorithm_interface.py

该文件包含以下各数据处理类的基类。

#### 文件 k_means.py

基于K-MEANS算法的数据处理类。

#### 文件 logistic_regression.py

基于逻辑回归算法的数据处理类。

#### 文件 random_forest.py

基于随机森林算法的数据处理类。

#### 文件 svm.py

基于支持向量机算法的数据处理类。

#### 文件 voting_classifier.py

基于以上算法模型的投票分类器类。

### 文件 main.py

包含用于展示和测试的代码。

## 项目逻辑

### 数据预处理

数据预处理一般是在基类`AlgorithmInterface`中完成的，其中包括数据加载与数据连续化等工作。

#### 数据加载

数据加载在函数`load_data()`中完成，具体代码如下所示。

首先按照用户行为进行分类，其中，字典值为0表示行为正常，1为攻击性行为。

随后，加载字段名称、训练数据和测试样例，以供随后使用。

```python
import pandas

def load_data(self):
    labels_2_dict = {'normal': 0, 'attack': 1, 'satan': 1, 'smurf': 1, 'spy': 1, 'teardrop': 1,
                     'warezclient': 1, 'warezmaster': 1, 'unknown': 1,
                     'back': 1, 'buffer_overflow': 1, 'ftp_write': 1, 'guess_passwd': 1, 'imap': 1,
                     'ipsweep': 1, 'land': 1, 'loadmodule': 1, 'multihop': 1, 'neptune': 1,
                     'nmap': 1, 'perl': 1, 'phf': 1, 'pod': 1, 'portsweep': 1, 'rootkit': 1, 'mailbomb': 1,
                     'apache2': 1, 'processtable': 1, 'mscan': 1, 'saint': 1, 'httptunnel': 1, 'snmpgetattack': 1,
                     'snmpguess': 1, 'sendmail': 1, 'ps': 1, 'xsnoop': 1, 'named': 1, 'xterm': 1, 'worm': 1,
                     'xlock': 1, 'sqlattack': 1, 'udpstorm': 1}

    columns_name_path = "KDD99/Field Names.csv"
    with open(columns_name_path, "r") as file:
        for line in file:
            key, data_type = line.split(",")
            self.columns[key] = data_type.replace("\n", "")

    train_data_path = "KDD99/KDDTrain+.csv"
    data = pandas.read_csv(train_data_path)
    self.train_data = data.iloc[:, 0:-2]
    self.train_label = data.iloc[:, -2]
    self.train_data.columns = self.columns.keys()
    self.train_label.replace(labels_2_dict, inplace=True)

    test_data_path = "KDD99/KDDTest+.csv"
    data = pandas.read_csv(test_data_path)
    self.test_data = data.iloc[:, 0:-2]
    self.test_label = data.iloc[:, -2]
    self.test_data.columns = self.columns.keys()
    self.test_label.replace(labels_2_dict, inplace=True)
```

#### 数据连续化

由于该模型仅可处理连续数据，故需要将数据集中的字符类型数据（如`SF`、`REJ`等）转换为浮点或整数类型。操作在函数`convert_symbolic_feature_into_continuous()`完成。

```python
def convert_symbolic_feature_into_continuous(self):
    for key in self.columns.keys():
        if self.columns[key] != "symbolic":
            continue

        category = dict()
        for i, symbolic in enumerate(self.train_data[key].unique()):
            category[symbolic] = i

        self.train_data[key].replace(category, inplace=True)
        self.test_data[key].replace(category, inplace=True)
        del category
```

### 模型建立

本项目使用了三个独立模型和一个融合了这三个模型的投票器。每个模型由一个类所管理，且均派生于基类`AlgorithmInterface`。

#### 随机森林模型

该模型位于类`RandomForestAlgorithm`中。

在随机森林模型的建立过程中，首先设置树的个数，也就是模拟器的个数`n_estimators`，然后设施树的最大深度`max_depth`，随后设置超参数`hyper_parameter_grid`。

最后，在设置交叉验证生成器、并行任务数、随机状态等参数后，调用`self.classifier.fit`函数进行训练。训练后的模型保存在`self.classifier`中。

```python
def train_phase(self):
    random_forest = RandomForestClassifier()

    n_estimators = [100]

    max_depth = [10, 20]

    hyper_parameter_grid = {'n_estimators': n_estimators,
                            'max_depth': max_depth}

    self.classifier = RandomizedSearchCV(estimator=random_forest,
                                         param_distributions=hyper_parameter_grid,
                                         cv=4, n_iter=1,
                                         scoring='roc_auc',
                                         n_jobs=-1, verbose=2,
                                         return_train_score=True,
                                         random_state=42)

    self.classifier.fit(self.train_data, self.train_label)
```

#### 支持向量机模型

该模型位于类`SVMAlgorithm`中。

支持向量机模型的建立过程中，首先设置标准化和支持向量分类器的管道，随后设置超参数`hyper_parameter_grid`。

最后，和上述模型相似，在设置交叉验证生成器、并行任务数、随机状态等参数后，调用`self.classifier.fit`函数进行训练。训练后的模型保存在`self.classifier`中。

```python
def train_phase(self):
    pipe_svc = Pipeline([('scl', StandardScaler()),
                         ('clf', SVC(random_state=1))])

    param_range = [0.0001, 0.001]
    hyper_parameter_grid = {
        'clf__C': param_range,
        'clf__gamma': param_range,
        'clf__kernel': ['linear', 'rbf']
    }

    self.classifier = RandomizedSearchCV(estimator=pipe_svc,
                                         param_distributions=hyper_parameter_grid,
                                         cv=4, n_iter=5,
                                         scoring='roc_auc',
                                         n_jobs=-1, verbose=2,
                                         return_train_score=True,
                                         random_state=42)

    self.classifier.fit(self.train_data, self.train_label)
```

#### 逻辑回归模型

该模型位于类`LogisticRegressionAlgorithm`中。

支持向量机模型的建立过程中，首先设置标准化、主成分分析和支持向量分类器的管道，随后设置超参数`hyper_parameter_grid`。

最后，和上述模型相似，在设置交叉验证生成器、并行任务数、随机状态等参数后，调用`self.classifier.fit`函数进行训练。训练后的模型保存在`self.classifier`中。

```python
def train_phase(self):
    pipe_logistic_regression = Pipeline([('sc', StandardScaler()),
                                        ('pca', PCA(n_components=2)),
                                        ('clf', LogisticRegression(random_state=1))
                                         ])

    param_range = [0.0001, 0.001]
    hyper_parameter_grid = {
        'clf__C': param_range,
    }

    self.classifier = RandomizedSearchCV(estimator=pipe_logistic_regression,
                                         param_distributions=hyper_parameter_grid,
                                         cv=4, n_iter=5,
                                         scoring='roc_auc',
                                         n_jobs=-1, verbose=2,
                                         return_train_score=True,
                                         random_state=42)

    self.classifier.fit(self.train_data, self.train_label)
```

#### 投票器模型

该模型位于类`VotingClassifier3`中，但可使用`ml_algorithms.VotingClassifier`进行调用。

由于在实验的测试过程中，上述三种模型的准确率很难超过80%，所以，我们决定对这三种模型进行融合。经分析，我们选择了投票器来进行融合。

在融合时，我们需要把上述三种模型的分类器作为参数传入到`VotingClassifier`中，然后调用`self.classifier.fit`函数进行训练。训练后的模型保存在`self.classifier`中。

```python
self.classifier = VotingClassifier(estimators=[
    ('rfa', rfa.classifier),
    ('svma', svma.classifier),
    ('lra', lra.classifier)
])

self.classifier.fit(self.test_data, self.test_label)
```

### 模型

#### 模型建立

可用如下代码加载并训练前文所述的模型。

```python
from ml_algorithms import (
    KMeansAlgorithm,
    RandomForestAlgorithm,
    SVMAlgorithm,
    LogisticRegressionAlgorithm,
    VotingClassifier
)

rfa = RandomForestAlgorithm()
rfa.run()

svma = SVMAlgorithm()
svma.run()

lra = LogisticRegressionAlgorithm()
lra.run()

vclf = VotingClassifier(rfa, svma, lra)
vclf = vclf.run()
```

#### 模型使用

为实现对单条数据的预测，我们已经建立了如下的函数进行预测.

```python
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
```

以下为使用方法举例。在此例中，我们模仿了一个聚类中的多条数据。

```python
# 在数据中取出1000条，模仿一组聚类，统计预测结果
for i in range(1, 1000):
    print('————————————————————')
    real_predict(rfa.test_data[i:i+1], rfa, svma, lra, vclf)
```

## 实验效果

在三个独立模型的测试中，经过多次调参测试，能达到的最高准确率均在76%左右，其结果如下：

随机森林模型结果：

```text
准确度: 0.767589
精确度: 0.811208
召回率: 0.792542
```
支持向量机模型结果：

```text
准确度: 0.769985
精确度: 0.815569
召回率: 0.795448
```
逻辑回归模型结果：

```text
准确度: 0.753216
精确度: 0.803483
召回率: 0.779955
```

但是在基于投票的分类器下，准确率大幅提高到约96%。实验结果如下：

```text
准确度: 0.958522
```

## 实验结论

总之，该模型在单个独立数据下已有准确率接近`P = 96%`的较好分类效果。另外，在与基于K-Means的无监督学习的网络用户行为分析算法的结合过程中，能够根据聚类中的多条数据联合判断，能够达到更高的识别准确率，能够达到实验效果。