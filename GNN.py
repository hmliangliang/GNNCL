#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 15:59
# @Author  : liangliang
# @Email   : shuliangxu@tencent.com
# @File    : GNN.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import time
import s3fs
import os
import pandas as pd
#os.system("pip install dgl-cu101 dglgo -f https://data.dgl.ai/wheels/repo.html")
os.system("pip install dgl") #CPU版本
os.system("python -m dgl.backend.set_default_backend \"tensorflow\"")
#os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true" #使用CPU版本注释此命令
import dgl
import dgl.nn as nn
import datetime


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
        n = len(data) #列表的长度
        s3fs.S3FileSystem = S3FileSystemPatched
        fs = s3fs.S3FileSystem()
        start = time.time()
        for i in range(n):
            with fs.open(self.output_path + 'pred_{}.csv'.format(int(i/args.file_max_num)), mode="a") as resultfile:
                # data = [line.decode('utf8').strip() for line in data.tolist()]
                #pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
                line = "{},{},{}\n".format(data[i][0],data[i][1],data[i][2])
                resultfile.write(line)
        cost = time.time() - start
        print("write is finish. write {} lines with {:.2f}s".format(len(data), cost))

class Net(keras.Model):
    def __init__(self,input_feat,output_feat1=150,output_feat2=100,output_feat=64):
        super(Net, self).__init__()
        self.cov1 = nn.GraphConv(input_feat,output_feat1,allow_zero_in_degree=True)
        self.cov2 = nn.SAGEConv(output_feat1,output_feat2,aggregator_type="mean")
        self.cov3 = keras.layers.Dense(output_feat)
    def call(self, inputs, g, training=None, mask=None):
        h = self.cov1(g,inputs)
        h = tf.nn.softmax(h)
        h = self.cov2(g,h)
        h = tf.nn.elu(h)
        h = self.cov3(h)
        h = tf.nn.elu(h)
        return h

class MLP(keras.Model):
    def __init__(self,output_feat1=150,output_feat2=100,output_feat=64):
        super(MLP, self).__init__()
        self.cov1 = keras.layers.Dense(output_feat1)
        self.cov2 = keras.layers.Dense(output_feat2)
        self.cov3 = keras.layers.Dense(output_feat)
    def call(self, feat, training=None, mask=None):
        h = self.cov1(feat)
        h = tf.nn.elu(h)
        h = self.cov2(h)
        h = tf.nn.elu(h)
        h = self.cov3(h)
        return h

def Loss(data, embed, embed_mlp, args):
    loss = 0
    N = embed_mlp.shape[0]#样本的个数
    num = data.shape[0]#边的个数
    for i in range(num):
        n = int(data[i,0])
        m = int(data[i,1])
        if data[i,2] == 0:
            loss = loss - 1 / num * tf.reduce_sum(embed[n, :] * embed[m, :]) / (
                        tf.norm(embed[n, :], 2) * tf.norm(embed[m, :], 2))
        elif data[i,2] == 1:
            loss = loss - 2 / num * tf.reduce_sum(embed[n, :] * embed[m, :]) / (
                        tf.norm(embed[n, :], 2) * tf.norm(embed[m, :], 2))
        else:
            loss = loss - 3 / num * tf.reduce_sum(embed[n, :] * embed[m, :]) / (
                        tf.norm(embed[n, :], 2) * tf.norm(embed[m, :], 2))
        #进行负采样
        seq_num = np.random.randint(0,N,args.neg_num)
        for j in seq_num:#计算负样本的相似度
            loss = loss + 1 / (args.neg_num * num) * tf.reduce_sum((0.8*embed[n, :] + 0.2*embed[j, :]) * embed[j, :]) / (
                    tf.norm(0.8*embed[n, :] + 0.2*embed[j, :], 2) * tf.norm(embed[j, :], 2))
            loss = loss + 1 / (args.neg_num * num) * tf.reduce_sum((0.8*embed[m, :] + 0.2*embed[j, :]) * embed[j, :]) / (
                    tf.norm(0.8*embed[m, :] + 0.2*embed[j, :], 2) * tf.norm(embed[j, :], 2))
    #对比学习损失
    for i in range(N):
        #正样本间的损失
        loss = loss - 1 / n * tf.reduce_sum(embed[i, :] * embed_mlp[i, :]) / (
                tf.norm(embed[i, :], 2) * tf.norm(embed_mlp[i, :], 2))
        seq_num = np.random.randint(0, n, args.neg_num)
        for j in seq_num:  # 计算负样本的相似度
            loss = loss + 1 / (args.neg_num * n) * tf.reduce_sum(
                (0.8 * embed[i, :] + 0.2 * embed_mlp[j, :]) * embed_mlp[j, :]) / (
                           tf.norm(0.8 * embed[i, :] + 0.2 * embed_mlp[j, :], 2) * tf.norm(embed_mlp[j, :], 2))
    #对比学习特征标准差最大化
    dim = embed_mlp.shape[1]
    loss = loss - 1 / dim * tf.reduce_sum(tf.math.reduce_std(embed, 0))
    loss = loss - 1 / dim * tf.reduce_sum(tf.math.reduce_std(embed_mlp, 0))
    #不同维度之间的相似度尽可能地小
    for i in range(dim):
        seq_num = np.random.randint(0, dim, args.neg_num)
        for j in seq_num:  # 计算负样本的相似度
            loss = loss + 1 / (args.neg_num * dim) * tf.reduce_sum(embed[:,i]*embed_mlp[:,j])/(tf.norm(embed[:,i], 2)*tf.norm(embed[:,j], 2))
    return loss

def train(args):
    #读取属性特征信息
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        data_attr = pd.concat([data_attr,pd.read_csv("s3://" + file, sep=',', header=None)], axis=0) #读取数据
    data_attr = tf.convert_to_tensor(data_attr.values, dtype=tf.float32)  # 读取节点的属性特征数据
    #定义图神经网络
    #读取数据,并重新对节点进行编号
    '''读取s3fs数据部分'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)  # 读取边结构数据
    if args.env == "train":#第一次训练,创建模型
        net = Net(data_attr.shape[1], args.hidden_num1, args.hidden_num2, args.output_dim)
        net_mlp = MLP(args.hidden_num1, args.hidden_num2, args.output_dim)
    else:#利用上一次训练好的模型，进行增量式训练
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net"
        os.system(cmd)
        net = keras.models.load_model("./JK_trained_model_net", custom_objects={'tf': tf}, compile=False)

        cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net_mlp"
        os.system(cmd)
        net_mlp = keras.models.load_model("./JK_trained_model_net_mlp", custom_objects={'tf': tf}, compile=False)
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    before_net = net
    before_mlpnet = net_mlp
    before_loss = 2 ** 32
    #读取图数据
    print("开始构建图! {}".format(datetime.datetime.now()))
    g = dgl.graph((data.iloc[:,0].to_list(),data.iloc[:,1].to_list()),idtype=tf.int32,num_nodes=data_attr.shape[0])
    g = dgl.add_self_loop(g)
    print("图构建完成! {}".format(datetime.datetime.now()))
    stop_num = 0
    for epoch in range(args.epoch):
        with tf.GradientTape() as tape:
            embed = net(data_attr,g, training=True)
            print("epoch:{} net embedding输出完成! {}".format(epoch,datetime.datetime.now()))
            embed_mlp = net_mlp(data_attr)
            print("epoch:{} net_mlp embedding输出完成! {}".format(epoch,datetime.datetime.now()))
            print("epoch:{} embed大小:{}  embed_mlp大小:{}  {}".format(epoch,embed.shape,embed_mlp.shape,datetime.datetime.now()))
            loss = Loss(data.values, embed,embed_mlp,args)#数据的第3列为类标签
            print("epoch:{} loss计算完成! {}".format(epoch,datetime.datetime.now()))
        gradients = tape.gradient(loss, net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, net.trainable_variables))
        print("epoch:{} 梯度优化完成! {}".format(epoch,datetime.datetime.now()))
        if loss > before_loss:
            before_loss = loss
            before_net = net
            before_mlpnet = net_mlp
            stop_num = 0
        else:
            stop_num = stop_num + 1
            if stop_num >= args.stop_num:
                print("Stop early!")
            break
        print("The epoch:{} Loss:{} The best loss:{}  {}".format(epoch,loss,before_loss,datetime.datetime.now()))
    #保存图神经网络模型
    flag = True
    if flag:
        net = before_net
        net_mlp = before_mlpnet
        #保存teacher net
        net.save("./JK_trained_model_net", save_format="tf")
        cmd = "s3cmd put -r ./JK_trained_model_net " + args.model_output
        os.system(cmd)
        #保存student net
        net_mlp.save("./JK_trained_model_net_mlp", save_format="tf")
        cmd = "s3cmd put -r ./JK_trained_model_net_mlp " + args.model_output
        os.system(cmd)
        return flag
    else:
        flag = False
        return flag
    print("模型保存完毕! {}".format(datetime.datetime.now()))

def test(args):
    # 装载训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net"
    os.system(cmd)
    net = keras.models.load_model("./JK_trained_model_net", custom_objects={'tf': tf}, compile=False)
    # 读取属性特征信息
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        data_attr = pd.concat([data_attr, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)  # 读取数据
    data_attr = tf.convert_to_tensor(data_attr.values, dtype=tf.float32)  # 读取数据
    '''读取s3fs边数据部分'''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data = pd.DataFrame()
    count = 0
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)  # 读取边结构数据
    g = dgl.graph((data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()), idtype=tf.int32, num_nodes=data_attr.shape[0])
    g = dgl.add_self_loop(g)
    embed = net(data_attr,g)
    #embed = tf.concat([data_attr,embed],axis=1)
    return embed
