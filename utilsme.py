import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as spp
import math
import pandas as pd
from collections import Counter
from sklearn.utils import check_array



"""
	Utility functions to handle early stopping and mixed droupout and mixed liner.
"""

class EarlyStopping:
    """
    EarlyStopping（早停机制）：
    ---------------------------------
    当模型在验证集上的性能（准确率或损失）长时间没有提升时，
    自动提前停止训练，以防止过拟合。
    """

    def __init__(self, patience=10):
        """
        初始化函数
        :param patience: 容忍验证集性能不提升的次数，超过则停止训练
        """
        self.patience = patience   # 连续多少次性能没变好就停止
        self.counter = 0           # 当前已经连续几次没有提升
        self.best_score = None     # 记录目前为止最好的验证集分数（或准确率）
        self.early_stop = False    # 是否触发早停的标志
        self.best_epoch = None     # 记录最优结果出现的 epoch（轮次）

    def step(self, acc, model, epoch):
        """
        每一轮训练结束后调用一次，用于判断是否要停止训练。
        :param acc: 本轮验证集上的准确率或性能指标
        :param model: 当前模型（用于保存参数）
        :param epoch: 当前是第几轮训练（方便记录最佳 epoch）
        :return: 布尔值 -> 是否应该停止训练
        """
        score = acc  # 将传入的准确率作为当前评估分数

        # 如果还没有记录过 best_score（说明是第一轮）
        if self.best_score is None:
            self.best_score = score       # 当前分数设为最优分数
            self.best_epoch = epoch       # 记录当前 epoch
            self.save_checkpoint(model)   # 保存当前模型参数

        # 如果当前分数比历史最优分数低（模型变差了）
        elif score < self.best_score:
            self.counter += 1             # 连续“未提升”次数 +1
            # 如果连续未提升次数 >= 允许的 patience，触发早停
            if self.counter >= self.patience:
                self.early_stop = True

        # 如果当前分数高于历史最优分数（模型有提升）
        else:
            self.best_score = score       # 更新最优分数
            self.best_epoch = epoch       # 更新最优轮次
            self.save_checkpoint(model)   # 保存当前模型参数
            self.counter = 0              # 连续未提升计数清零

        # 返回是否触发早停（True 表示要提前终止训练）
        return self.early_stop

    def save_checkpoint(self, model):
        """
        当验证集分数提升时，保存当前模型参数。
        这里固定将参数保存为 'es_checkpoint.pt' 文件。
        """
        torch.save(model.state_dict(), 'es_checkpoint.pt')


# =========================================================
# 1️⃣ 稀疏矩阵版 Dropout —— 适用于输入为稀疏张量的情况
# =========================================================
class SparseDropout(nn.Module):
    def __init__(self, p):
        """
        初始化函数
        :param p: dropout 概率（比如 0.3 表示随机丢弃 30% 的元素）
        """
        super().__init__()
        self.p = p  # 保存dropout概率

    def forward(self, input):
        """
        前向传播函数：对稀疏张量的非零元素执行 Dropout 操作。
        """
        # coalesce()：标准化稀疏张量（去重索引、合并重复项）
        input_coal = input.coalesce()

        # 从稀疏张量中取出所有非零值并执行dropout
        # F.dropout仅作用在这些值上，不改变索引位置
        drop_val = F.dropout(input_coal._values(), self.p, self.training)

        # 用原来的索引和新的值重新构造一个稀疏张量返回
        return torch.sparse.FloatTensor(input_coal._indices(), drop_val, input.shape)


# =========================================================
# 2️⃣ 混合版 Dropout —— 自动判断稀疏或稠密输入
# =========================================================
class MixedDropout(nn.Module):
    def __init__(self, p):
        """
        初始化函数
        :param p: dropout 概率
        """
        super().__init__()
        # 对稠密张量使用普通 Dropout
        self.dense_dropout = nn.Dropout(p)
        # 对稀疏张量使用自定义的 SparseDropout
        self.sparse_dropout = SparseDropout(p)

    def forward(self, input):
        """
        前向传播：根据输入类型自动选择合适的 Dropout 实现。
        """
        if input.is_sparse:
            # 稀疏输入 → 使用自定义稀疏 Dropout
            return self.sparse_dropout(input)
        else:
            # 稠密输入 → 使用普通 Dropout
            return self.dense_dropout(input)


# =========================================================
# 3️⃣ 混合版 Linear 层 —— 支持稀疏/稠密输入的线性变换
# =========================================================
class MixedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        """
        初始化函数
        :param in_features: 输入维度大小
        :param out_features: 输出维度大小
        :param bias: 是否使用偏置项
        """
        super().__init__()
        self.in_features = in_features   # 输入特征数
        self.out_features = out_features # 输出特征数

        # 定义权重矩阵 (形状：[in_features, out_features])
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))

        # 若需要偏置，则定义偏置向量
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            # 若不需要偏置，则注册一个空参数（方便模型保存时结构完整）
            self.register_parameter('bias', None)

        # 初始化权重和偏置
        self.reset_parameters()

    def reset_parameters(self):
        """
        参数初始化函数：
        使用Kaiming He初始化方法为权重赋值，
        保持网络在初始阶段的梯度稳定。
        """
        # 由于权重矩阵的维度定义与PyTorch默认不同，这里mode使用'fan_out'
        nn.init.kaiming_uniform_(self.weight, mode='fan_out', a=math.sqrt(5))

        # 若存在偏置项，则计算其初始化范围
        if self.bias is not None:
            # 自动计算fan_in和fan_out（输入/输出通道数）
            _, fan_out = nn.init._calculate_fan_in_and_fan_out(self.weight)

            # 计算均匀分布边界 [-bound, bound]
            bound = 1 / math.sqrt(fan_out)

            # 在该范围内均匀采样初始化偏置
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        """
        前向传播函数：计算 y = XW + b
        能够自动区分稀疏和稠密输入的乘法方式。
        """
        # 若没有偏置项
        if self.bias is None:
            if input.is_sparse:
                # 稀疏输入 → 使用 torch.sparse.mm 进行稀疏矩阵乘法
                res = torch.sparse.mm(input, self.weight)
            else:
                # 稠密输入 → 普通矩阵乘法
                res = input.matmul(self.weight)
        else:
            # 若有偏置项
            if input.is_sparse:
                # 稀疏输入 → 使用 torch.sparse.addmm：计算 bias + XW
                res = torch.sparse.addmm(self.bias.expand(input.shape[0], -1),
                                         input, self.weight)
            else:
                # 稠密输入 → 使用 torch.addmm：计算 bias + XW
                res = torch.addmm(self.bias, input, self.weight)

        # 返回结果张量
        return res

    def extra_repr(self):
        """
        用于打印层信息（在print(model)时显示更直观）
        """
        return 'in_features={}, out_features={}, bias={}'.format(
                self.in_features, self.out_features, self.bias is not None)


# =========================================================
# 4️⃣ 辅助函数：将稀疏 / 稠密矩阵 转换为 PyTorch 张量
# =========================================================

def sparse_matrix_to_torch(X):
    """
    将 scipy 的稀疏矩阵 (CSR/CSC) 转换为 PyTorch 的稀疏张量。

    参数：
        X: scipy.sparse 矩阵对象

    返回：
        torch.sparse.FloatTensor 稀疏张量
    """
    coo = X.tocoo()  # 转为 COO 格式（方便提取坐标和值）
    indices = np.array([coo.row, coo.col])  # 取出非零元素的位置 (2 × nnz)

    # 构造 PyTorch 稀疏张量：
    #   indices -> 非零位置坐标
    #   values  -> 非零元素的值
    #   shape   -> 原矩阵形状
    return torch.sparse.FloatTensor(
        torch.LongTensor(indices.astype(np.float32)),  # 坐标索引
        torch.FloatTensor(coo.data),                   # 非零值
        coo.shape                                      # 形状
    )


def matrix_to_torch(X):
    """
    将 numpy / scipy 矩阵统一转换为 PyTorch 张量。
    自动判断稀疏 or 稠密。
    """
    if spp.issparse(X):           # 如果是稀疏矩阵
        return sparse_matrix_to_torch(X)
    else:                         # 如果是普通矩阵
        return torch.FloatTensor(X)

# =========================================================
# 5️⃣ 计算误分类代价（cost-sensitive learning 的核心）
# =========================================================

def misclassification_cost(y_true, y_pred, cost_table):
    """
    根据真实标签和预测标签，从代价表中查找每个样本对应的误分类代价。

    参数：
        y_true : 真实标签 (array-like)
        y_pred : 预测标签 (array-like)
        cost_table : DataFrame，包含三列 ['row', 'column', 'cost']
                     row = 预测类别，column = 真实类别，cost = 对应代价值
    返回：
        numpy 数组，每个样本的 cost 值。
    """
    # 构建一个包含预测类和真实类的 DataFrame
    df = pd.DataFrame({'row': y_pred, 'column': y_true})

    # 按(row, column)字段与代价表合并，左连接保留所有样本
    df = df.merge(cost_table, how='left', on=['row', 'column'])

    # 返回每个样本对应的代价值
    return df['cost'].values


# =========================================================
# 6️⃣ 构建代价矩阵（cost matrix）
# =========================================================

SET_COST_MATRIX_HOW = ('uniform', 'inverse', 'log1p-inverse')

def _set_cost_matrix(y, how: str = 'inverse'):
    """
    根据数据分布生成类别代价矩阵。

    参数：
        y : 标签数组 (1D)
        how : 代价计算策略，可选：
              'uniform'        -> 全部 cost=1，不考虑不平衡
              'inverse'        -> 用类别样本数比例的倒数
              'log1p-inverse'  -> log(1 + ratio) 形式平滑比例

    返回：
        cost_matrix : ndarray (n_classes × n_classes)
    """
    # 获取所有类别及其对应的编码
    classes_, _y_encoded = np.unique(y, return_inverse=True)

    # 构造类别 -> 编号的映射表，如 {0:0, 1:1, 2:2}
    _encode_map = {c: np.where(classes_ == c)[0][0] for c in classes_}

    # 统计每个类别的样本数量
    origin_distr_ = dict(Counter(_y_encoded))  # e.g., {0: 1000, 1: 50}
    classes, origin_distr = _encode_map.values(), origin_distr_

    cost_matrix = []

    # 遍历每个预测类别，构建代价矩阵的每一行
    for c_pred in classes:
        # 计算每个实际类别的代价比例 = N_pred / N_actual
        cost_c = [
            origin_distr[c_pred] / origin_distr[c_actual]
            for c_actual in classes
        ]
        # 对角线（预测正确的情况）代价=1
        cost_c[c_pred] = 1
        cost_matrix.append(cost_c)

    # 根据选择的模式决定最终矩阵形式
    if how == 'uniform':
        return np.ones_like(cost_matrix)          # 全部设为1
    elif how == 'inverse':
        return cost_matrix                        # 使用比例倒数
    elif how == 'log1p-inverse':
        return np.log1p(cost_matrix)               # log(1+x) 平滑版本
    else:
        raise ValueError(
            f"When 'cost_matrix' is string, it should be"
            f" in {SET_COST_MATRIX_HOW}, got {how}."
        )


# =========================================================
# 7️⃣ 将代价矩阵展开为表格形式（方便查找）
# =========================================================

def cost_table_calc(cost_matrix):
    """
    把矩阵形式的 cost_matrix 转换成表格形式的 DataFrame，
    每一行对应矩阵中的一个元素 (row, column, cost)。
    参数：
        cost_matrix : numpy.ndarray (n_classes × n_classes)

    返回：
        pandas.DataFrame，列名 ['row', 'column', 'cost']
    """
    table = np.empty((0, 3))  # 初始化空表

    # 遍历 cost_matrix 中的每个元素
    for (x, y), value in np.ndenumerate(cost_matrix):
        # np.ndenumerate 返回索引(x,y)与对应值value
        table = np.vstack((table, np.array([x, y, value])))

    # 转换为 pandas.DataFrame，便于后续合并使用
    return pd.DataFrame(table, columns=['row', 'column', 'cost'])


# =========================================================
# 8️⃣ 验证代价矩阵是否合法（维度检查）
# =========================================================

def _validate_cost_matrix(cost_matrix, n_classes):
    """
    检查输入的 cost_matrix 是否满足基本要求：
    - 是二维矩阵；
    - 不含 NaN 或 inf；
    - 形状正确 (n_classes × n_classes)
    """
    # 调用 sklearn 的检查工具
    cost_matrix = check_array(cost_matrix,
        ensure_2d=True, allow_nd=False,
        force_all_finite=True)

    # 若形状不符合要求，则抛出异常
    if cost_matrix.shape != (n_classes, n_classes):
        raise ValueError(
            "When 'cost_matrix' is array-like, it should"
            " be of shape = [n_classes, n_classes],"
            " got shape = {0}".format(cost_matrix.shape)
        )

    # 返回验证后的矩阵
    return cost_matrix


def calc_group_difficulty(logits, g, device):
    """
    [动态计算] 团伙难度权重 G_{g(v)}
    
    逻辑：
    1. 获取欺诈节点的预测误差 (1 - p_t)
    2. 按 comp_id (团伙ID) 聚合，计算每个团伙中欺诈节点的平均误差
    3. 将团伙平均误差映射回该团伙的所有节点
    """
    labels = g.ndata['label'].to(device)
    comp_id = g.ndata['comp_id'].to(device)
    
    # 1. 计算 p_t (模型对真实标签的预测概率)
    # logits shape: [N, 2]
    probs = F.softmax(logits, dim=1)
    # gather: 取出真实标签对应的概率值
    pt = probs.gather(1, labels.view(-1, 1)).view(-1)
    
    # 2. 计算误差 (Error = 1 - p_t)
    error = 1.0 - pt
    
    # 3. 仅针对欺诈节点计算团伙难度
    # (正常节点贡献的难度设为0，只关注欺诈节点的识别情况)
    is_fraud = (labels == 1).float()
    fraud_error = error * is_fraud
    
    # --- 聚合计算 (Group Aggregation) ---
    # 获取团伙总数 (最大ID + 1)
    num_comps = comp_id.max().item() + 1
    
    # 分子：每个团伙的 fraud_error 总和
    group_error_sum = torch.zeros(num_comps, device=device)
    group_error_sum.index_add_(0, comp_id, fraud_error)
    
    # 分母：每个团伙的 fraud 节点数量
    group_fraud_count = torch.zeros(num_comps, device=device)
    group_fraud_count.index_add_(0, comp_id, is_fraud)
    
    # 避免除以 0 (对于没有 fraud 的纯正常团伙，分母设为 1，分子为 0，结果为 0)
    group_fraud_count = torch.clamp(group_fraud_count, min=1.0)
    
    # 得到每个团伙的平均误差 [num_comps]
    group_avg_error = group_error_sum / group_fraud_count
    
    # 4. 映射回节点 (Broadcast back to nodes)
    # 节点 i 的团伙难度 = group_avg_error[comp_id[i]]
    G_val = group_avg_error[comp_id]
    
    return G_val


def calc_camouflage_weight(logits, g, args, device):
    """
    [动态计算] 伪装难例权重
    
    定义:
        D_v = Het_v * (1 - p_t)
        W_camo = 1 + lambda_D * D_v
        
    逻辑：
        只有当节点处于高异质性环境(Het大) 且 模型预测错误(1-pt大) 时，
        D_v 才会大，从而触发高权重惩罚。
    """
    labels = g.ndata['label'].to(device)
    het = g.ndata['heterophily'].to(device) # 静态异质性 (Het_v)
    
    # 1. 计算 p_t (模型对真实标签的预测概率)
    probs = F.softmax(logits, dim=1)
    pt = probs.gather(1, labels.view(-1, 1)).view(-1)
    
    # 2. 计算 D_v (Camouflage Score)
    # 修正后的公式：不再包含 gamma 指数
    d_v_score = het * (1.0 - pt)
    
    # 3. 计算最终乘数 W_camo
    w_camo_val = 1.0 + args.lambda_D * d_v_score
    
    return w_camo_val


def calc_has_gnn_loss(logits, g, args, device):
    """
    [核心] HAS-GNN 最终加权损失函数
    
    L = - sum( W_total * log(p_t) ) / N_train
    W_total = alpha_class * W_struct * W_group * W_camo
    """
    labels = g.ndata['label'].to(device)
    
    # --- 1. 获取/计算各部分权重 ---
    
    # A. 类别权重 (静态，已在 data_process 中计算好，含 Log 平滑)
    w_class = g.ndata['class_weight'].to(device)
    
    # B. 结构权重 (静态，已在 data_process 中计算好，相对 K-Core)
    # W_struct = 1 + lambda_I * struct_score
    w_struct = 1.0 + args.lambda_I * g.ndata['struct_score'].to(device)
    
    # C. 团伙权重 (动态，基于当前 logits)
    # W_group = 1 + lambda_G * G_val
    G_val = calc_group_difficulty(logits, g, device)
    w_group = 1.0 + args.lambda_G * G_val
    
    # D. 伪装权重 (动态，基于当前 logits)
    # W_camo = 1 + lambda_D * (Het * (1-pt))
    w_camo = calc_camouflage_weight(logits, g, args, device)
    
    # --- 2. 组合总权重 ---
    # Element-wise 乘积：四个维度的权重叠加
    w_total = w_class * w_struct * w_group * w_camo
    
    # --- 3. 计算标准 CE Loss (不平均，reduction='none') ---
    # F.cross_entropy = log_softmax + nll_loss
    # 返回 shape [N]
    ce_loss = F.cross_entropy(logits, labels, reduction='none')
    
    # --- 4. 加权 ---
    weighted_loss = w_total * ce_loss
    
    # --- 5. 只对训练集求平均 ---
    # 确保只反向传播训练节点的 Loss
    train_mask = g.ndata['train_mask'].to(device)
    final_loss = weighted_loss[train_mask].mean()
    
    return final_loss