GNNCL文件表示使用单进程进行推理计算，而GNNCLmultithreading使用多线程进行推理计算。在7980264个节点的图中，计算出所有节点的嵌入向量，单进程需要19天12小时51分10秒。在采用多线程加速后，9313160个节点的图，计算出所有节点的嵌入向量，只需要22小时59分22秒，加速效果十分明显。

GNNCL算法推理时，需要把完整的图传入内存，因此I/O操作频繁，不适合用多进程加速，实验效果显示，采用多进程加速方案后，实际加速效果还不如单进程运行方案。

GNNCL算法需要安装dgl>=0.8.1, 因为dgl.khop_in_subgraph函数只有在0.8.1及之后的版本中才有,如果dgl的版本小于0.8.1,库内无dgl.khop_in_subgraph函数,导致程序运行报错。
