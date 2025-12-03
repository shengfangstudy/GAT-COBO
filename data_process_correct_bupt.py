import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info


def BUPT_process():
    """
    读取 BUPT 数据集原始文件（特征、标签、边），
    进行对齐与预处理后，构建 DGL 图并保存为 BUPT_tele.bin / BUPT_tele.pkl。
    """

    # ===================== 1. 读取原始文件 =====================
    # feature.shape = (N, 1+F)：第一列是原始节点 ID，后面 F 列是特征
    feature = np.genfromtxt("./data/BUPT/TF.features",
                            dtype=np.dtype(str),
                            delimiter=' ')
    # labels.shape = (M, 2)：第一列原始节点 ID，第二列是类别（可能 M >= N）
    labels = np.genfromtxt("./data/BUPT/TF.labels",
                           dtype=np.dtype(str),
                           delimiter=' ')
    # edges.shape = (E, 2)：每行一条边，两个都是原始节点 ID
    edges = np.genfromtxt("./data/BUPT/TF.edgelist",
                          dtype=np.dtype(str),
                          delimiter=' ')

    # ===================== 2. 特征处理 =====================
    # 2.1 提取节点 ID（与特征矩阵一一对应）
    # feat_ids[i] 就是第 i 行特征所属的原始节点 ID
    feat_ids = feature[:, 0].astype(np.int32)          # shape: (N,)

    # 2.2 提取并标准化特征 (N, F)
    features = feature[:, 1:].astype(np.float32)       # 去掉第一列 ID
    norm_features = StandardScaler().fit_transform(features)

    # ===================== 3. 标签对齐（按 node_id 映射） =====================
    # 3.1 从 labels 中取出 ID 和标签
    label_ids = labels[:, 0].astype(np.int32)          # shape: (M,)
    label_vals = labels[:, 1].astype(np.int32)         # shape: (M,)

    # 3.2 构建「原始节点 ID -> 标签」映射表
    id2label = {nid: lab for nid, lab in zip(label_ids, label_vals)}

    # 3.3 按 feat_ids 的顺序取出每个节点的标签，确保「特征和标签一一对应」
    #     如果某个 feat_id 在 labels 里找不到，会抛异常，方便你排查数据问题
    label_extract = np.array(
        [id2label[nid] for nid in feat_ids],
        dtype=np.int32
    )   # shape: (N,)

    # ===================== 4. 节点重新编号 old_id -> new_id =====================
    # 这里我们让所有出现过的 feature 节点的编号变成 0,1,...,N-1
    # new_id = 节点在 feat_ids 里的行号
    node_new_num = np.arange(len(feat_ids), dtype=np.int32)   # 0..N-1
    # 建「原始 ID -> 新 ID」映射表
    id2new = {old_id: new_id for new_id, old_id in enumerate(feat_ids)}

    # ===================== 5. 处理边，过滤无效节点并映射到新编号 =====================
    edges_int = edges.astype(np.int32)           # (E, 2)
    src_raw = edges_int[:, 0]
    dst_raw = edges_int[:, 1]

    # 5.1 只保留「两端节点都在 feat_ids 里」的边
    #     简单起见，用 np.isin 做过滤
    mask_src_in = np.isin(src_raw, feat_ids)
    mask_dst_in = np.isin(dst_raw, feat_ids)
    valid_mask = mask_src_in & mask_dst_in

    src_valid = src_raw[valid_mask]
    dst_valid = dst_raw[valid_mask]

    # 5.2 把原始 ID 映射成 0..N-1 的新 ID
    #     用列表推导配合 id2new 映射
    src_mapped = np.array([id2new[s] for s in src_valid], dtype=np.int32)
    dst_mapped = np.array([id2new[d] for d in dst_valid], dtype=np.int32)

    # 合并成 (E_valid, 2) 的边列表
    new_all_edges = np.stack([src_mapped, dst_mapped], axis=1)  # shape: (E_valid, 2)

    # ===================== 6. 构建无向图 + 自环 =====================
    # 6.1 只保留上三角 (src < dst)，避免重复边
    triu_edges = new_all_edges[new_all_edges[:, 0] < new_all_edges[:, 1]]

    # 6.2 为每条边加一个反向边，得到对称的无向图边集合
    symmetry_edges = triu_edges[:, [1, 0]]              # 交换列 (dst, src)
    homo_graph_edges = np.concatenate((triu_edges, symmetry_edges), axis=0)

    # 6.3 对所有边去重，避免重复边
    homo_graph_edges_unique = np.unique(homo_graph_edges, axis=0)

    # 6.4 为每个节点加自环 (i, i)
    selfloop_edges = np.stack([node_new_num, node_new_num], axis=1)   # (N, 2)
    homo_graph_edges_unique_selfloop = np.concatenate(
        (homo_graph_edges_unique, selfloop_edges),
        axis=0
    )

    # 6.5 再去重一次，得到最终边集合
    homo_graph = np.unique(homo_graph_edges_unique_selfloop, axis=0)  # (E_final, 2)

    # ===================== 7. 构建 DGL 图 =====================
    src_id = homo_graph[:, 0]
    dst_id = homo_graph[:, 1]

    graph = dgl.graph(
        (torch.tensor(src_id, dtype=torch.int64),
         torch.tensor(dst_id, dtype=torch.int64))
    )

    # 转成 torch 张量，特征与标签已经和节点顺序对齐
    graph.ndata['feat'] = torch.tensor(norm_features, dtype=torch.float32)   # (N, F)
    graph.ndata['label'] = torch.tensor(label_extract, dtype=torch.int64)    # (N,)

    # ===================== 8. 保存到文件 =====================
    save_graphs('./data/BUPT_tele_correct.bin', graph, {'labels': graph.ndata['label']})
    save_info('./data/BUPT_tele.pkl', {'num_classes': 3})
    print('The BUPT dataset is successfully generated! ')


if __name__ == "__main__":
    # 仅处理 BUPT 数据集（Sichuan 的在 TelcomFraudDataset 里）
    BUPT_process()
