#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 下午1:56
# @Author  : Joselynzhao
# @Email   : zhaojing17@forxmail.com
# @File    : drawer_kf.py
# @Software: PyCharm
# @Desc    :


import os
import codecs
import os.path as osp
import math
import matplotlib.pyplot as plt
import numpy as np
from  pathlib import  Path


'''本文件只在本地跑.'''
class drawer_kf(): #处理单一文件.
    def __init__(self,file_name,dataset):
        '''
        :dataset : 数据集名称
        :param file_path: 形如 atm/pro1_t2
        '''
        self.file_name = file_name.split('/')[-1]
        self.dataset = dataset
        self.file_path = osp.join(self.dataset,file_name,'kf.txt')
        self.save_path = osp.join(self.dataset,'figure_out','kf') # 这个是固定不变的
        if not Path(self.save_path).exists():
            print(self.save_path+'is not exist, creating now!')
            os.makedirs(self.save_path)  # 可生成多级目录]
        self.lable_item = 'raw_label_pre,t_label_pre,kf_label_pre'.split(',')
        self.select_item = 'raw_select_pre,t_label_pre,kf_select_pre'.split(',')
        self.label_pre = []
        self.select_pre = []

        self.get_datas()
        self.drawer()

    def get_datas(self):
        self.data = []
        file = ''
        try:
            file = open(self.file_path,'r')
        except FileNotFoundError:
            print("the file named {} is not found!".format(self.file_path))
        infos = file.readlines()
        if len(infos)!=0:
            for info in infos:
                info_f = info.strip('\n').split(' ')[3:]  # 不要step
                info_f = list(map(float, [dd.strip('%') for dd in info_f]))
                info_f = np.array(info_f)
                self.data.append(info_f)
            self.data = np.array(self.data)
            h,w = self.data.shape
            print("h={},w={}".format(h,w))
            self.label_pre = np.array(self.data[:,[0,3]] if w==5 else self.data[:,[0,3,5]])
            self.select_pre = np.array(self.data[:,[2,4]] if w==5 else self.data[:,[2,4,6]])
            print(self.label_pre.shape)
            print(self.select_pre.shape)
        else:
            print("the file named {} is empty!".format(self.file_path))

    def drawer(self):
        h,w = self.data.shape
        raw,col = 1,2
        unit_size = 4
        dpi = 300
        hspace, wspace = 0.3,0.3
        plt.figure(figsize=(col * unit_size * 1.5, raw * unit_size), dpi=dpi)
        plt.suptitle(self.dataset + '-' + self.file_name)
        plt.subplots_adjust(hspace=hspace, wspace=wspace)
        print("开始绘图")
        plt.subplot(1,2,1) # 画label_pre
        x = np.linspace(1,h,h)
        for i in range(self.label_pre.shape[1]):
            y = self.label_pre[:,i]
            max_point = np.argmax(y)
            plt.annotate(str(y[max_point]),xy = (max_point,y[max_point]))
            plt.plot(x,y,label = self.lable_item[i],marker = 'o')
        plt.xlabel('step')
        plt.xticks(np.arange(1,h+1,1))
        plt.ylabel('value(%)')
        plt.title('label_pre')
        plt.legend()

        plt.subplot(1, 2, 2)  # 画label_pre
        x = np.linspace(1, h, h)
        for i in range(self.select_pre.shape[1]):
            y = self.select_pre[:,i]
            max_point = np.argmax(y)
            plt.annotate(str(y[max_point]), xy=(max_point, y[max_point]))
            plt.plot(x, y, label=self.select_item[i], marker='o')
        plt.xlabel('step')
        plt.xticks(np.arange(1, h + 1, 1))
        plt.ylabel('value(%)')
        plt.title('select_pre')
        plt.legend()

        plt.savefig(osp.join(self.save_path,self.file_name),bbox_inches='tight')


if __name__ =='__main__':

    datasets = ['duke','DukeMTMC-VideoReID','market1501','mars']
    file_name = 'atm/pro1_t2'
    D = drawer_kf(file_name,datasets[2])