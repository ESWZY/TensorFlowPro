from random import shuffle

import numpy
import tensorflow as tf
from numpy import array

from .ml_algorithm_interface import AlgorithmBaseInterface


class KMeansAlgorithm(AlgorithmBaseInterface):
    """
    Run K-Means algorithm.
    """

    def __init__(self):
        super(KMeansAlgorithm, self).__init__()
        self.DATA_PATH = "KDD99/data_10000.csv"  # 未添加label的数据集

        # tcp -> 1
        # icmp - > 2
        # udp -> 3

        # http -> 1
        # whois -> 2
        # vmnet -> 3
        # 其他的按照字母表顺序从4排到59

        # SF -> 1
        # S0 -> 2
        # S1 -> 3
        # S2 -> 4
        # REJ -> 5
        # RSTR -> 6

        # normal. -> 1
        # back. -> 2
        # neptune. -> 3
        # smurf. -> 4

        super(KMeansAlgorithm, self).__init__()

    def handle(self, data_matrix):
        row_data = data_matrix.size / len(data_matrix[0])
        column_data = len(data_matrix[0])

        for i in range(int(row_data)):
            for j in range(column_data):
                if j == 0:
                    data_matrix[i][j] = data_matrix[i][j] * 0.001
                elif j == 1:
                    data_matrix[i][j] = data_matrix[i][j] * 0.001
                elif j == 27:
                    data_matrix[i][j] = data_matrix[i][j] * 0.01
                elif j == 28:
                    data_matrix[i][j] = data_matrix[i][j] * 0.01
        return data_matrix

    def load_data(self):
        f = open(self.DATA_PATH, "rb")
        data_matrix = numpy.loadtxt(self.DATA_PATH, delimiter=",", skiprows=0)
        f.close()

        row_data = data_matrix.size / len(data_matrix[0])
        self.column_data = len(data_matrix[0])
        self.row_data = int(row_data)

        self.data = self.handle(data_matrix)

    def KMeansCluster(self, vectors, noofclusters):
        """
        基于TensorFlow的K-Means算法

        :param vectors: n*k的二维数组，n是向量的个数，k是向量的维度
        :param noofclusters: 分簇簇头的个数
        :return:
        """

        noofclusters = int(noofclusters)
        assert noofclusters < len(vectors)

        # 获取向量的维度
        dim = len(vectors[0])

        # Will help select random centroids from among the available vectors
        # 将所有向量存入list中方便随机选取簇头
        vector_indices = list(range(len(vectors)))

        #############################################################change at 7/10/2018
        shuffle(vector_indices)
        # tf.random_shuffle(vector_indices)

        graph = tf.Graph()

        with graph.as_default():

            # SESSION OF COMPUTATION

            self.sess = tf.compat.v1.Session()

            centroids = [tf.Variable((vectors[vector_indices[i]]))
                         for i in range(noofclusters)]

            centroid_value = tf.compat.v1.placeholder("float64", [dim])
            cent_assigns = []
            for centroid in centroids:
                cent_assigns.append(tf.compat.v1.assign(centroid, centroid_value))

            assignments = [tf.Variable(0) for i in range(len(vectors))]

            assignment_value = tf.compat.v1.placeholder("int32")
            cluster_assigns = []
            for assignment in assignments:
                cluster_assigns.append(tf.compat.v1.assign(assignment,
                                                           assignment_value))

            # 创建一个用于计算均值的变量
            mean_input = tf.compat.v1.placeholder("float", [None, dim])
            # 用于获取输入的向量组，并从0维开始计算平均值
            mean_op = tf.reduce_mean(mean_input, 0)

            # 用于计算欧氏距离的两个变量
            v1 = tf.compat.v1.placeholder("float", [dim])
            v2 = tf.compat.v1.placeholder("float", [dim])

            euclid_dist = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(v1, v2), 2)))

            ##构建一个根据欧氏距离分配向量到簇的变量
            centroid_distances = tf.compat.v1.placeholder("float", [noofclusters])
            cluster_assignment = tf.argmin(centroid_distances, 0)

            # 变量初始化器
            init_op = tf.compat.v1.global_variables_initializer()

            # 初始化所有变量
            self.sess.run(init_op)

            noofiterations = 10
            for iteration_n in range(noofiterations):

                # 根据上一次计算得到的簇头计算每个向量应该被分配到哪个簇

                # 遍历每个向量
                for vector_n in range(len(vectors)):
                    vect = vectors[vector_n]
                    # 计算这个向量到每个簇头的欧式距离
                    distances = [self.sess.run(euclid_dist, feed_dict={
                        v1: vect, v2: self.sess.run(centroid)})
                                 for centroid in centroids]
                    # 将分配节点信息和欧氏距离当做输入
                    assignment = self.sess.run(cluster_assignment, feed_dict={
                        centroid_distances: distances})
                    #
                    # 分配适当的状态变量
                    self.sess.run(cluster_assigns[vector_n], feed_dict={
                        assignment_value: assignment})

                # 根据上面得到的变量，计算质心的位置
                for cluster_n in range(noofclusters):
                    # Collect all the vectors assigned to this cluster
                    # 获取所有属于同簇的向量
                    assigned_vects = [vectors[i] for i in range(len(vectors))
                                      if self.sess.run(assignments[i]) == cluster_n]
                    # 计算出新的簇头节点坐标
                    new_location = self.sess.run(mean_op, feed_dict={
                        mean_input: array(assigned_vects)})
                    # 分配适当的变量
                    self.sess.run(cent_assigns[cluster_n], feed_dict={
                        centroid_value: new_location})

            centroids = self.sess.run(centroids)
            assignments = self.sess.run(assignments)

            return centroids, assignments

    def run(self):
        c, a = self.KMeansCluster(self.data, 4)

        for i in c:
            print(i)
        # print(a)

        cnt = [0, 0, 0, 0, 0]
        for i in a:
            if i < 4:
                cnt[i] += 1
            else:
                cnt[4] += 1
        for i in range(4):
            print("quantity of num{} is ".format(i) + str(cnt[i]))
        print("Clustering finish!")

    def save(self):
        saver = tf.compat.v1.train.Saver()
        save_path = saver.save(self.sess,"model_success")
        print ("Model saved in path :",save_path)
