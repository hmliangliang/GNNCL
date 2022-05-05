#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 15:59
# @Author  : liangliang
# @File    : GNN.py
# @Software: PyCharm
import os
import tensorflow as tf

#设置多核并行
os.environ["OMP_NUM_THREADS"] = "19" # 1为一个核，设置为5的时候，系统显示用了10个核，不太清楚之间的具体数量关系
tf.config.threading.set_intra_op_parallelism_threads(9)
tf.config.threading.set_inter_op_parallelism_threads(9)

import numpy as np
import tensorflow.keras as keras
import time
import s3fs
import random
import pandas as pd
#os.system("pip install dgl-cu101 dglgo -f https://data.dgl.ai/wheels/repo.html") #若使用GPU环境需要安装dgl的GPU版本
os.system("pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html") #若使用CPU环境需要安装dgl的CPU版本
os.environ['DGLBACKEND'] = "tensorflow"
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import dgl
from dgl import nn as nn
import datetime

#设置随机种子点
#random.seed(921208)
#np.random.seed(921208)
#tf.random.set_seed(921208)
#os.environ['PYTHONHASHSEED'] = "921208"
#os.environ['TF_DETERMINISTIC_OPS'] = '1' #设置GPU随机种子点

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

#执行batch_normalization操作
def batch_normalization(data,epsilon = 0.00001):
    mean, var = tf.nn.moments(data, axes=[0])
    size = data.shape[1]
    scale = tf.Variable(tf.ones([size]))
    shift = tf.Variable(tf.zeros([size]))
    data = tf.nn.batch_normalization(data, mean, var, shift, scale, epsilon)
    return data


class Net(keras.Model):
    def __init__(self,input_feat,output_feat1=150,output_feat2=100,output_feat=64):
        super(Net, self).__init__()
        self.cov1 = nn.GraphConv(input_feat,output_feat1,norm='both', weight=True, bias=True,allow_zero_in_degree=True)
        self.cov2 = nn.SAGEConv(output_feat1,output_feat2,aggregator_type="mean")
        self.cov3 = keras.layers.Dense(output_feat)

    def call(self, g, inputs, training=None, mask=None):
        h = self.cov1(g,inputs)
        h = tf.nn.elu(h)
        #h = batch_normalization(h)
        h = self.cov2(g,h)
        h = tf.nn.elu(h)
        #h = batch_normalization(h)
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
        #h = batch_normalization(h)
        h = self.cov2(h)
        h = tf.nn.elu(h)
        #h = batch_normalization(h)
        h = self.cov3(h)
        return h

def Loss(data, embed, embed_mlp, args):
    loss = 0
    N = embed_mlp.shape[0]#样本的个数
    num = data.shape[0]#边的个数
    for i in range(num):
        n = int(data[i,0])
        m = int(data[i,1])
        if data[i,2] == 1: #1:玩家-对局玩家 2:玩家-自身俱乐部 3:玩家-对局玩家俱乐部
            loss = loss - 1 / num * tf.reduce_sum(embed[n, :] * embed[m, :]) / (
                        tf.norm(embed[n, :], 2) * tf.norm(embed[m, :], 2)+0.000001)
        elif data[i,2] == 2:
            loss = loss - 1.2 / num * tf.reduce_sum(embed[n, :] * embed[m, :]) / (
                        tf.norm(embed[n, :], 2) * tf.norm(embed[m, :], 2)+0.000001)
        else:
            loss = loss - 1.2 / num * tf.reduce_sum(embed[n, :] * embed[m, :]) / (
                        tf.norm(embed[n, :], 2) * tf.norm(embed[m, :], 2)+0.000001)
        #进行负采样
        seq_num = np.random.randint(0,N,max(args.neg_num-4,1))
        for j in seq_num:#计算负样本的相似度
            loss = loss + 1 / (args.neg_num * num) * tf.reduce_sum((0.8*embed[n, :] + 0.2*embed[j, :]) * embed[j, :]) / (
                    tf.norm(0.8*embed[n, :] + 0.2*embed[j, :], 2) * tf.norm(embed[j, :], 2)+0.000001)
            loss = loss + 1 / (args.neg_num * num) * tf.reduce_sum((0.8*embed[m, :] + 0.2*embed[j, :]) * embed[j, :]) / (
                    tf.norm(0.8*embed[m, :] + 0.2*embed[j, :], 2) * tf.norm(embed[j, :], 2)+0.000001)
    #对比学习损失
    for i in range(N):
        #正样本间的损失
        loss = loss - 1 / n * tf.reduce_sum(embed[i, :] * embed_mlp[i, :]) / (
                tf.norm(embed[i, :], 2) * tf.norm(embed_mlp[i, :], 2)+0.000001)
        seq_num = np.random.randint(0, n, args.neg_num)
        for j in seq_num:  # 计算负样本的相似度
            loss = loss + 1 / (args.neg_num * n) * tf.reduce_sum(
                (0.8 * embed[i, :] + 0.2 * embed_mlp[j, :]) * embed_mlp[j, :]) / (
                           tf.norm(0.8 * embed[i, :] + 0.2 * embed_mlp[j, :], 2) * tf.norm(embed_mlp[j, :], 2)+0.000001)
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

        "装载数据模型"
        net = Net(data_attr.shape[1], args.hidden_num1, args.hidden_num2, args.output_dim)
        checkpoint_path = "./JK_trained_model_net/JK_trained_model_net.pb"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        net.load_weights(latest)
        print("Teacher Model weights load!")

        #cmd = "s3cmd get -r  " + args.model_output + "JK_trained_model_net_mlp"
        #os.system(cmd)
        net_mlp = keras.models.load_model("./JK_trained_model_net_mlp", custom_objects={'tf': tf}, compile=False)
        print("Student Model weights load!")
    #读取图数据
    g = dgl.graph((data.iloc[:,0].to_list(),data.iloc[:,1].to_list()),num_nodes=data_attr.shape[0]).int()
    #将原始图装入gpu
    #g = g.to('gpu:0') #用gpu训练需要注释此行代码
    g.ndata['feat'] = data_attr #分配节点属性,若在gpu环境中，则把数据放入gpu中
    g.edata['label'] = tf.reshape(tf.convert_to_tensor(data.iloc[:,2],dtype=tf.int32),(-1,1)) #分配边的类标签,若在gpu环境中，则把数据放入gpu中
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    for batch_num in range(args.batch_num):#多次采样子图
        before_net = net
        before_mlpnet = net_mlp
        before_loss = 2 ** 32
        #对大规模图进行采样
        g_sub, _ = dgl.khop_in_subgraph(g,np.random.randint(0,data_attr.shape[0],args.nodes_num).tolist(),k=args.k_hop,store_ids=False)
        while g_sub.num_nodes() >= args.subgraph_nodes_max_num or g_sub.num_edges() >= args.subgraph_edges_max_num:
            g_sub, _ = dgl.khop_in_subgraph(g, np.random.randint(0, data_attr.shape[0], args.nodes_num).tolist(),
                                            k=args.k_hop, store_ids=False)
        print("batch_num:",batch_num,"  g_sub=",g_sub)
        stop_num = 0
        for epoch in range(args.epoch):
            with tf.GradientTape() as tape:
                embed = net(g_sub,g_sub.ndata["feat"], training=True)
                embed_mlp = net_mlp(g_sub.ndata["feat"], training=True)
                edge_data = tf.concat([tf.reshape(g_sub.edges()[0],(-1,1)),tf.reshape(g_sub.edges()[1],(-1,1)),tf.reshape(g_sub.edata["label"],(-1,1))],axis=1)
                loss = Loss(edge_data, embed,embed_mlp,args)#数据的第3列为类标签
            print("batch_num:{} epoch:{} loss:{} {}".format(batch_num,epoch,loss.numpy(),datetime.datetime.now()))
            gradients = tape.gradient(loss, net.trainable_variables)
            optimizer.apply_gradients(zip(gradients, net.trainable_variables))
            if loss < before_loss:
                before_loss = loss
                before_net = net
                before_mlpnet = net_mlp
                stop_num = 0
            else:
                stop_num = stop_num + 1
                if stop_num >= args.stop_num:
                    print("Stop early!")
                    break
            print("batch_num:{} The epoch:{} Loss:{} The best loss:{}  {}".format(batch_num,epoch,loss,before_loss,datetime.datetime.now()))
    #保存图神经网络模型
    flag = True
    if flag:
        net = before_net
        net_mlp = before_mlpnet
        #保存teacher net
        #net.summary()
        net.save_weights("./JK_trained_model_net/JK_trained_model_net.pb", save_format="tf")
        print("net已保存!")
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
    #net = keras.models.load_model("./JK_trained_model_net", custom_objects={'tf': tf}, compile=False)
    # 读取属性特征信息
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        data_attr = pd.concat([data_attr, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)  # 读取数据
    data_attr = tf.convert_to_tensor(data_attr.values, dtype=tf.float32)  # 读取数据
    "装载数据模型"
    net = Net(data_attr.shape[1], args.hidden_num1, args.hidden_num2, args.output_dim)
    checkpoint_path = "./JK_trained_model_net/JK_trained_model_net.pb.index"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    net.load_weights(latest)
    print("Model weights load!")
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
    g = dgl.graph((data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()), num_nodes=data_attr.shape[0]).int()
    g.ndata["feat"] = data_attr
    N = g.num_nodes()
    embed = tf.zeros((1, args.output_dim + 1))  # 预先定义张量
    #对子图进行取样
    for i in range(N):
        # 对大规模图进行采样
        print("一共有{}个节点，当前正在处理第{}个节点 {}".format(N,i,datetime.datetime.now()))
        g_sub, _ = dgl.khop_in_subgraph(g, i, k=args.k_hop, store_ids=True)
        k = args.k_hop
        flag = True
        while g_sub.num_nodes() >= args.subgraph_nodes_max_num or g_sub.num_edges() >= args.subgraph_edges_max_num:
            print("i={} k={}采样的子图过大,可能会造成内存溢出，正在重新采样子图!{}".format(i,k,datetime.datetime.now()))
            k = k - 1
            if k<1:#当度为1的子图仍然超过规模，则不计算当前节点的嵌入
                flag = False
                break
            else:
                g_sub, _ = dgl.khop_in_subgraph(g, i,k=k, store_ids=True)
                print("采样到新的子图为:{}".format(g_sub))
        if flag == False:
            print("第{}个节点图采样跳过!{}".format(i,datetime.datetime.now()))
            continue
        embed_tmp = net(g_sub,g_sub.ndata["feat"])
        j = tf.where(g_sub.ndata[dgl.NID] == i)[0,0] #获取中心节点在原始图中真实的编号与当前子图中编号间的映射关系
        embed = tf.concat([embed,tf.concat([tf.constant([[i]],dtype=tf.float32),tf.reshape(embed_tmp[j,:],(1,-1))],axis=1)],axis=0)
    #第一行是用来声明tensorflow进行占位，返回前需要去掉
    if embed.shape[0] == 1:#所有节点都没有被采样子图,此时第一行第一列的元素为-1.0
        embed = tf.concat([tf.constant([[-1.0]],dtype=tf.float32),tf.zeros((1,args.output_dim))],axis=1)
    else:
        embed = embed[1::, :]
    return embed #注意的是embed的第一列为节点的id编号