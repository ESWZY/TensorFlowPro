# -*- coding: utf-8 -*-

from ml_algorithms import KMeansAlgorithm

DATA_PATH = "data_10000.csv"  # 未添加label的数据集
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

if __name__ == '__main__':
    km = KMeansAlgorithm()
    km.run()
