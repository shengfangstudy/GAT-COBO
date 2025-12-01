baselines文件是基线模型，没碰过
data
  Sichuan文件夹是四川数据集，
    all_feat_with_label是提取好的特征，第一列是加密手机号，顺序和原有的user文件相同
    node_adj_sparse是图邻接矩阵，GAT-COBO的构图
  Sichuan_tele.bin是数据预处理后得到的图dgl数据（由data_process文件得到）
  Sichuan_tele.pkl是只有类别（由data_process文件得到）
environment文件是运行环境


training_records_sichuan_0是在原有代码上加了训练监控的训练记录
  时间文件是每一次模型训练的结果记录
    experiment_summary是模型的参数
    layer_1_metrics是第一个弱分类器的曲线，Loss（应用权重了）、AUC、F1、Recall（Macro）
    layer_2_metrics是第一个弱分类器的曲线，Loss（应用权重了）、AUC、F1、Recall（Macro）
  all_training_logs详细日志，记录了每一次训练的信息

es_checkpoint.pt是应用早停技术的最佳模型，不使用早停，就没有
main_bupt.py是在原有代码基础上跑通，并设置好BUPT数据集的运行参数
main_sichuan.py是原有代码跑通-->加上监控机制
main_sichuan0.py是在原有代码基础上跑通，并设置好Sichuan数据集的运行参数
main.py是最原始的代码+跑通，但是运行时得指定参数，其它代码运行不需要指定参数
model.py是原始的代码+跑通
README.md是作者的
utils是工具类