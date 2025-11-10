from dgl.nn import GATConv
from utils import MixedDropout, MixedLinear
import torch.nn as nn
from numpy import random
import dgl

class GAT_COBO(nn.Module):
    """
    GAT-COBO 模型主体。
    论文：GAT-COBO: Cost-Sensitive Graph Neural Network for Anomaly Detection

    整体思路：
    - 同时学习 “节点特征视角” 和 “图结构视角” 的两种分类器；
    - 前者通过线性层 (MixedLinear) 提取节点自身特征；
    - 后者通过多层 GATConv 学习节点之间的拓扑关系；
    - 最终结合两种结果，形成 Cost-Sensitive 异常检测模型。
    """

    def __init__(self,
                 g,                 # DGL 图对象
                 num_layers,        # GAT 层数
                 in_dim,            # 输入特征维度
                 num_hidden,        # 隐层维度
                 num_classes,       # 输出类别数
                 heads,             # 每层的注意力头数（list）
                 activation,        # 激活函数，如 ReLU / ELU
                 dropout,           # 输入特征 dropout 比例
                 dropout_adj,       # 邻接 dropout 比例
                 feat_drop,         # GAT 层内部的特征 dropout
                 attn_drop,         # GAT 层内部的注意力 dropout
                 negative_slope,    # LeakyReLU 负斜率
                 residual):         # 是否使用残差连接
        super(GAT_COBO, self).__init__()

        # ============================================================
        # ① 特征变换部分 (Feature Transformation)
        # ============================================================

        # 第一层：MixedLinear (支持稀疏输入)
        fcs = [MixedLinear(in_dim, num_hidden, bias=False)]
        # 第二层：普通线性层，将 hidden -> class logits
        fcs.append(nn.Linear(num_hidden, num_classes, bias=False))

        # 将这两层加入模块列表，方便在 forward 中循环使用
        self.fcs = nn.ModuleList(fcs)

        # 用于正则化（只对第一层）
        self.reg_params = list(self.fcs[0].parameters())

        # 定义特征 dropout
        if dropout == 0:
            self.dropout = lambda x: x  # 若设置为0，直接返回输入
        else:
            self.dropout = MixedDropout(dropout)

        # 定义邻接 dropout（增强鲁棒性）
        if dropout_adj == 0:
            self.dropout_adj = lambda x: x
        else:
            self.dropout_adj = MixedDropout(dropout_adj)

        # 激活函数（可换成 F.elu、F.leaky_relu 等）
        self.act_fn = nn.ReLU()


        # ============================================================
        # ② GAT 层部分 (Graph Attention Network)
        # ============================================================

        self.g = g                        # 图对象
        self.num_layers = num_layers      # GAT 层数
        self.gat_layers = nn.ModuleList() # 用列表保存多层 GATConv
        self.activation = activation      # 激活函数

        # ---- 输入层：in_dim -> hidden_dim ----
        self.gat_layers.append(GATConv(
            in_dim,               # 输入特征维度
            num_hidden,           # 输出维度
            heads[0],             # 注意力头数
            feat_drop,            # 特征 dropout
            attn_drop,            # 注意力 dropout
            negative_slope,       # LeakyReLU 参数
            False,                # 不使用残差
            self.activation,      # 激活函数
            bias=False))          # 无偏置项

        # ---- 隐藏层（如果有多层）----
        for l in range(1, num_layers):
            # 多头输出拼接，所以输入维度 = num_hidden * heads[l-1]
            self.gat_layers.append(GATConv(
                num_hidden * heads[l-1],
                num_hidden,
                heads[l],
                feat_drop,
                attn_drop,
                negative_slope,
                residual,          # 可选残差
                self.activation,
                bias=False))

        # ---- 输出层 ----
        # 输出层负责生成类别预测 logits，不再有激活函数
        self.gat_layers.append(GATConv(
            num_hidden * heads[-2],  # 输入维度
            num_classes,             # 输出类别数
            heads[-1],               # 注意力头数
            feat_drop,
            attn_drop,
            negative_slope,
            residual,
            None))                   # 输出层无激活函数


    # ============================================================
    # ③ transform_features：特征空间弱分类器
    # ============================================================
    def transform_features(self, x):
        """
        功能：
        - 对节点特征 x 做若干层线性变换 + 激活 + Dropout；
        - 相当于节点自身特征视角下的“弱分类器”（不考虑邻居信息）。

        输入：
            x: [num_nodes, in_dim] 节点特征矩阵

        输出：
            res: [num_nodes, num_classes] 每个节点的特征空间分类结果
        """

        # Step1：对输入特征施加 dropout，并通过第1层线性变换
        layer_inner = self.act_fn(self.fcs[0](self.dropout(x)))

        # Step2：如果有中间层，则依次线性变换 + 激活
        for fc in self.fcs[1:-1]:
            layer_inner = self.act_fn(fc(layer_inner))

        # Step3：最后一层 + 邻接 dropout + 激活
        res = self.act_fn(self.fcs[-1](self.dropout_adj(layer_inner)))

        # 返回节点级特征分类输出
        return res


    # ============================================================
    # ④ forward：完整前向传播
    # ============================================================
    def forward(self, inputs):
        """
        模型前向传播。
        同时计算两种分类输出 + 注意力矩阵。

        输入：
            inputs: 节点特征矩阵 X

        输出：
            logits_inter_GAT: 特征空间弱分类结果
            logits_inner_GAT: 图结构弱分类结果
            attention: 最后一层注意力权重
        """

        # (1) 先通过特征空间弱分类器（不考虑图结构）
        logits_inter_GAT = self.transform_features(inputs)

        # (2) 再通过多层 GATConv（考虑图结构）
        h = inputs
        for l in range(self.num_layers):
            # 每层 GATConv 输出多头拼接向量
            h = self.gat_layers[l](self.g, h).flatten(1)

        # (3) 最后一层 GATConv：输出 logits + attention 矩阵
        # logits_inner_GAT, attention = self.gat_layers[-1](self.g, h, True)
        logits_inner_GAT, attention = self.gat_layers[-1](self.g, h, get_attention=True)

        # (4) 返回三部分结果
        return logits_inter_GAT, logits_inner_GAT, attention