import os
import codecs
import os.path as osp
import math
import matplotlib.pyplot as plt
import numpy as np
from  pathlib import  Path


'''本文件只在本地跑.'''
class drawer01():
    def __init__(self,compare_path_list,save_name):
        # father = Path('/mnt/')
        # if father.exists():  # 是在服务器上
        #     print("正在服务器上运行")
        #     self.logs_dir = '/mnt/home/'  # 服务器
        # else:  # 本地
        #     self.logs_dir = '/home/joselyn/workspace/ATM_SERIES/'  # 本地跑用这个
        self.compare_path_list = compare_path_list
        self.save_name = save_name
        self.names = [(path.split('/')[-2])+'-tagper' if path.split('/')[-1]=='tagper' else  path.split('/')[-1] for path in self.compare_path_list]
        self.dataset = self.compare_path_list[0].split('/')[0]  # 第一个是数据集.
        self.save_path = osp.join(self.dataset,'figure_out',self.save_name) # 这个是固定不变的
        if not Path(self.save_path).exists():
            print(self.save_path+'is not exist, creating now!')
            os.makedirs(self.save_path)  # 可生成多级目录]

        desc_file = open(osp.join(self.save_path,'desc.txt'),'w')
        desc_file.write(self.save_name+'\n'+'compare_path:\n')
        for path in self.compare_path_list:
            desc_file.write(path+'\n')

        desc_file.close()
        self.item_name_P = 'mAP,Rank-1,Rank-5,Rank-10,Rank-20'.split(',')
        self.item_name_F = 'label_pre,select_pre'.split(',')



    def get_datas_F(self): # 获取特征相关的 label_pre和select_pre.
        # 正常情况下, 这两个数据是在dataf.txt 文件中的. 但是早期的训练.这个两个数据在data中.
        self.dataf = []
        for path in self.compare_path_list:
            file_data = []
            try:
                file = open(osp.join(path,'dataf.txt'),'r')
                seg = [1,3]
            except FileNotFoundError:
                file = open(osp.join(path,'data.txt'),'r')
                seg = [6,8]
            infos = file.readlines()
            for info in infos:
                info_f = info.strip('\n').split(' ')[seg[0]:seg[1]]  # 不要第一个 step
                info_f = list(map(float, [dd.strip('%') for dd in info_f]))
                info_f = np.array(info_f)
                file_data.append(info_f)
            self.dataf.append(file_data)
        self.dataf = np.array(self.dataf)
        print(self.dataf.shape)

    def get_datas_P(self): # 取性能数据.
        self.data = []
        for path in self.compare_path_list: # 对每一个实验去数据.
            # path = osp.join(self.logs_dir, path)
            file_data = []
            file = open(osp.join(path,'data.txt'),'r')
            infos = file.readlines()
            for info in infos:
                info_f = info.strip('\n').split(' ')[1:6]#不要step
                info_f = list(map(float, [dd.strip('%') for dd in info_f]))
                info_f = np.array(info_f)
                file_data.append(info_f)
            self.data.append(file_data)
        self.data = np.array(self.data)
        print(self.data.shape)

    def draw_Feat(self):
        self.get_datas_F()
        # 只有两个项目 label_pre 和 select_pre
        raw,col = 1,2
        self.__draw(self.dataf,self.item_name_F,raw,col,'Features')

    def draw_Perf(self): #只考虑perfermance的部分.
        self.get_datas_P()
        raw,col = 2,3
        png_name = 'Performance' if self.save_name != 'atm_vs_tagper' else '{}'.format(self.names[0])
        self.__draw(self.data,self.item_name_P,raw,col,png_name)

    def __draw(self,data,items,raw,col,fig_name,unit_size=4,hspace=0.3,wspace=0.3,dpi=300):
        plt.figure(figsize=(col*unit_size*1.5,raw*unit_size),dpi = dpi)
        plt.suptitle(self.dataset+'-'+self.save_name)
        plt.subplots_adjust(hspace=hspace,wspace=wspace)
        max_len = 0
        print("开始绘图")
        for idx, item_name in enumerate(items):
            print(idx,item_name)
            plt.subplot(raw,col,idx+1)
            # print('self.data',len(data))
            for idx_info,info in enumerate(data): #遍历所有的
                # print(info)
                # print(len(info))
                max_len = max(max_len, len(info))
                info = np.array(info)
                print(info.shape)
                max_point = np.argmax(info[:,idx])
                plt.annotate(str(info[max_point][idx]),xy=(max_point,info[max_point][idx]))
                x = np.linspace(1,len(info),len(info))
                plt.plot(x,info[:,idx],label=self.names[idx_info],marker='o')
            plt.xlabel('steps')
            # plt.xlim((0, 5))
            # plt.xlim()
            my_x_ticks = np.arange(1, max_len+1, 1)
            # my_y_ticks = np.arange(-2, 2, 0.3)
            plt.xticks(my_x_ticks)

            plt.ylabel('value(%)')
            plt.title(item_name)
            plt.legend()
        plt.savefig(osp.join(self.save_path,fig_name),bbox_inches='tight')


if __name__ =='__main__':
    analysis ={
        'baseline_EF10vsEF5':{
            'atm/0',
            'baseline/EF-5',
            # 'baseline/EF-10'
        },
        'EF10vsATM':{
            'atm/0',
            # 'baseline/EF-10',
            'atm/atm_0'
        },
        'atm_t_20_18_15':{
            'atm/atm_vs_0',
            'atm/atm_t18',
            'atm/atm_t15',
            'atm/0'
            # 'baseline/EF-10',
            # 'baseline/EF-5'
        },
        'atm_Ct_EF5':{
            'atm/atm_t20_EF5',
            'atm/atm_t18_EF5',
            'atm/atm_t15_EF5',
            'baseline/EF-5'
        },
        'atmkf_base_atm':{
            'atm/0',
            'atm/atm_t15',
            'atm/atmkf_t15'
        },
        'atm_vs_tagper':{
            'atm/pro1_t2',
            'atm/pro1_t2/tagper',
        },
        'atmpro1_vs_atm15and0':{
            'atm/pro1_t1',
            'atm/pro1_t2',
            'atm/atm_t15',
            'atm/0'
        }
    }
    datasets = ['duke','DukeMTMC-VideoReID','market1501','mars']
    save_name = 'atm_vs_tagper'
    compare_path = [osp.join(datasets[2],k) for k in analysis[save_name]]
    # compare_path = odatasets[1]
    print(compare_path)
    drawer = drawer01(compare_path,save_name)
    drawer.draw_Perf()
    if save_name!='atm_vs_tagper':
        drawer.draw_Feat()



