#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/2/28 17:06
# @Author  : liangliang
# @File    : execute.py
# @Software: PyCharm

import argparse
import GNN

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train')
    parser.add_argument("--lr", help="学习率", type=float, default=0.0001)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=10)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=60)
    parser.add_argument("--neg_num", help="负采样的数目", type=int, default=5)
    parser.add_argument("--hidden_num1", help="隐含层神经元的数目", type=int, default=150)
    parser.add_argument("--hidden_num2", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="隐含层神经元的数目", type=int, default=64)
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
        result = GNN.train(args, data=None)
        if result == 1:
            print("结果写入完成!")
        else:
            print("结果写入失败！")
    else:
        print("输入的环境参数错误,env只能为train或test!")