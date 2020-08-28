#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 下午4:38
# @Author  : Joselynzhao
# @Email   : zhaojing17@forxmail.com
# @File    : tst.py
# @Software: PyCharm
# @Desc    :
import  numpy as np

def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

a = np.array([-1,-2,-3,-4,5,6,8,9,10])
a = normalization(a)
print(a)
