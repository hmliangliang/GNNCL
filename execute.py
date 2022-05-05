#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 17:06
# @Author  : liangliang
# @File    : execute.py
# @Software: PyCharm

import argparse
import GNN
import s3fs
import time
import os
import multiprocessing
import math
import datetime

def multiprocessingWrite(file_number,data,output_path):
    print("开始写第{}个文件 {}".format(file_number,datetime.datetime.now()))
    n = len(data)  # 列表的长度
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    with fs.open(output_path + 'pred_{}.csv'.format(int(file_number)), mode="a") as resultfile:
        if n > 1:#说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:#说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    print("第{}个文件已经写入完成,写入数据的行数{} {}".format(file_number,n,datetime.datetime.now()))

class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )
class S3Filewrite:
    def __init__(self, args):
        super(S3Filewrite, self).__init__()
        self.output_path = args.data_output

    def write(self, data, args):
        #注意在此业务中data是一个二维list
        n_data = len(data) #数据的数量
        n = math.ceil(n_data/args.file_max_num) #列表的长度
        s3fs.S3FileSystem = S3FileSystemPatched
        pool = multiprocessing.Pool(processes=multiprocessing.cpu_count() - 1)
        start = time.time()
        for i in range(0,n,args.file_max_num):
            pool.apply_async(multiprocessingWrite, args=(i, data[i*args.file_max_num:min((i+1)*args.file_max_num,n_data)],self.output_path,))
        pool.close()
        pool.join()
        cost = time.time() - start
        print("write is finish. write {} lines with {:.2f}s".format(n_data, cost))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=5)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=10)
    parser.add_argument("--neg_num", help="负采样的数目", type=int, default=3)
    parser.add_argument("--nodes_num", help="采样子图的节点数目", type=int, default=1)
    parser.add_argument("--subgraph_nodes_max_num", help="采样出子图的最大节点数目", type=int, default=1000000)
    parser.add_argument("--subgraph_edges_max_num", help="采样出子图最大边的数目", type=int, default=2000000)
    parser.add_argument("--batch_num", help="采样子图的子图数目", type=int, default=100)
    parser.add_argument("--k_hop", help="采样子图的跳连数目", type=int, default=2)
    parser.add_argument("--hidden_num1", help="隐含层神经元的数目", type=int, default=150)
    parser.add_argument("--hidden_num2", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="隐含层神经元的数目", type=int, default=64)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=100000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置", type=str, default='s3://JK/models/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        result = GNN.train(args)
        if result:
            print("模型训练完成!")
        else:
            print("模型训练或保存失败！")
    elif args.env == "test":
        embed = GNN.test(args)
        if embed[0,0] == -1:
            print("图节点均没有被采样，无嵌入向量输入!")
        else:
            writer = S3Filewrite(args)
            writer.write(embed.numpy().tolist(), args)
    else:
        print("输入的环境参数错误,env只能为train或test!")