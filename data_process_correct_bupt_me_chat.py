import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info
import networkx as nx


def BUPT_process():
    """
    读取 BUPT 数据集原始文件（特征、标签、边），
    进行对齐与预处理后，构建带有 HAS-GNN 所需静态信息的 DGL 图：
    - comp_id: 连通分量编号
    - struct_score: 结构重要性（k-core，只有 fraud 结点有非零值）
    - heterophily: 异质性（fraud vs non-fraud）
    - class_weight: 类别逆频率 log 平滑权重
    """
    # ===================== 1. 读取原始文件 =====================
    feature = np.genfromtxt(
        "./data/BUPT/TF.features",
        dtype=np.dtype(str),
        delimiter=' '
    )
    labels = np.genfromtxt(
        "./data/BUPT/TF.labels",
        dtype=np.dtype(str),
        delimiter=' '
    )
    edges = np.genfromtxt(
        "./data/BUPT/TF.edgelist",
        dtype=np.dtype(str),
        delimiter=' '
    )

    # ===================== 2. 特征处理 =====================
    # feat_ids[i] 是第 i 行特征对应的原始节点 ID
    feat_ids = feature[:, 0].astype(np.int32)          # shape: (N,)
    N = len(feat_ids)
    node_new_num = np.arange(N, dtype=np.int32)        # 0..N-1

    # 特征矩阵并标准化
    features = feature[:, 1:].astype(np.float32)       # (N, F)
    norm_features = StandardScaler().fit_transform(features)

    # ===================== 3. 标签对齐 =====================
    label_ids = labels[:, 0].astype(np.int32)
    label_vals = labels[:, 1].astype(np.int32)         # 0/1/2 或 0/1/3

    id2label = {nid: lab for nid, lab in zip(label_ids, label_vals)}

    # 按 feat_ids 顺序抽取标签，确保 feat & label 对齐
    labels_np = np.array(
        [id2label[nid] for nid in feat_ids],
        dtype=np.int32
    )  # shape: (N,)

    print("=== BUPT 标签统计 ===")
    uniq, cnts = np.unique(labels_np, return_counts=True)
    for c, n_c in zip(uniq, cnts):
        print(f"  类别 {c}: {n_c} 个结点")
    print("==================================")

    # ===================== 4. 节点重新编号 old_id -> new_id =====================
    id2new = {old_id: new_id for new_id, old_id in enumerate(feat_ids)}

    # ===================== 5. 处理边：过滤 & 映射 & 无向 =====================
    edges_int = edges.astype(np.int32)
    src_raw = edges_int[:, 0]
    dst_raw = edges_int[:, 1]

    # 只保留两端都在 feat_ids 中的边
    mask_src_in = np.isin(src_raw, feat_ids)
    mask_dst_in = np.isin(dst_raw, feat_ids)
    valid_mask = mask_src_in & mask_dst_in

    src_valid = src_raw[valid_mask]
    dst_valid = dst_raw[valid_mask]

    # 映射到新编号 0..N-1
    src_mapped = np.array([id2new[s] for s in src_valid], dtype=np.int32)
    dst_mapped = np.array([id2new[d] for d in dst_valid], dtype=np.int32)

    # 只保留上三角，避免重复边
    edge_pairs = np.stack([src_mapped, dst_mapped], axis=1)  # (E_valid, 2)
    triu_edges = edge_pairs[edge_pairs[:, 0] < edge_pairs[:, 1]]

    # 无向化：加反向边，然后去重（仍然不含自环）
    symmetry_edges = triu_edges[:, [1, 0]]
    undirected_edges = np.concatenate([triu_edges, symmetry_edges], axis=0)
    undirected_edges_unique = np.unique(undirected_edges, axis=0)

    # 构造自环 (i, i) 并加入图用于 GNN
    selfloop_edges = np.stack([node_new_num, node_new_num], axis=1)
    homo_graph_edges = np.concatenate(
        [undirected_edges_unique, selfloop_edges],
        axis=0
    )
    homo_graph_edges = np.unique(homo_graph_edges, axis=0)   # 最终边集 (含自环)

    # ===================== 6. 构造原始无向邻接矩阵 adj_raw（无自环） =====================
    # adj_raw 只用于 comp_id / k-core / heterophily
    row = undirected_edges_unique[:, 0]
    col = undirected_edges_unique[:, 1]
    data = np.ones(len(row), dtype=np.float32)

    adj_raw = spp.coo_matrix((data, (row, col)), shape=(N, N))

    # ===================== 7. 连通分量 comp_id =====================
    print("Computing connected components...")
    n_comp, comp_labels = spp.csgraph.connected_components(
        adj_raw, directed=False, return_labels=True
    )
    comp_id = torch.from_numpy(comp_labels).long()
    print(f"共 {n_comp} 个连通子图")

    # ===================== 8. 结构重要性 struct_score (k-core) =====================
    print("Computing structural importance (k-core)...")
    G_nx = nx.from_scipy_sparse_array(adj_raw)  # 无向、无自环
    G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

    struct_score = torch.zeros(N, dtype=torch.float32)

    # 欺诈标签 == 1；快递员与普通用户均视为 non-fraud
    for cid in range(n_comp):
        nodes = np.where(comp_labels == cid)[0]
        if len(nodes) <= 1:
            continue

        sub_g = G_nx.subgraph(nodes)
        core_dict = nx.core_number(sub_g)
        if len(core_dict) == 0:
            continue

        cmax = max(core_dict.values())
        if cmax == 0:
            continue

        for v in nodes:
            if labels_np[v] == 1:  # 只有 fraud 节点有结构加权
                struct_score[v] = core_dict[v] / cmax

    # ===================== 9. 静态异质性 heterophily =====================
    print("Computing static heterophily...")

    # 邻居计数: 对每一类分别构造 indicator
    is_normal   = (labels_np == 0).astype(np.int32)
    is_fraud    = (labels_np == 1).astype(np.int32)
    is_courier  = (labels_np == 2).astype(np.int32)  # 或者 ==3，看数据集定义

    # 矩阵乘法统计邻居中属于某类的数量
    num_normal_neighbors  = adj_raw.dot(is_normal)
    num_fraud_neighbors   = adj_raw.dot(is_fraud)
    num_courier_neighbors = adj_raw.dot(is_courier)

    degree = np.array(adj_raw.sum(1)).flatten()
    degree[degree == 0] = 1  # 防止除零

    hetero = np.zeros(N, dtype=np.float32)

    # --- Fraud 节点：异质性 = （总 - fraud 邻居）/ 总
    fraud_mask = (labels_np == 1)
    hetero[fraud_mask] = (
        degree[fraud_mask] - num_fraud_neighbors[fraud_mask]
    ) / degree[fraud_mask]

    # --- Normal 节点：异质性 = （总 - normal 邻居）/ 总
    normal_mask = (labels_np == 0)
    hetero[normal_mask] = (
        degree[normal_mask] - num_normal_neighbors[normal_mask]
    ) / degree[normal_mask]

    # --- Courier 节点：异质性 = （总 - courier 邻居）/ 总
    courier_mask = (labels_np == 2)  # 若有 label=3 再 OR 一下
    hetero[courier_mask] = (
        degree[courier_mask] - num_courier_neighbors[courier_mask]
    ) / degree[courier_mask]

    heterophily = torch.from_numpy(hetero)

    # ===================== 10. 多类别 log-smooth 类别权重 =====================
    print("Computing class weights (multi-class log-smoothed)...")

    classes, counts = np.unique(labels_np, return_counts=True)
    C = len(classes)
    N_total = len(labels_np)

    cls2w = {}
    for c, n_c in zip(classes, counts):
        cls2w[c] = np.log(1.0 + N_total / (C * max(int(n_c), 1)))

    class_weight_np = np.array(
        [cls2w[y] for y in labels_np], dtype=np.float32
    )
    class_weight = torch.from_numpy(class_weight_np)

    # ===================== 11. 构建 DGL 图 =====================
    print("Building final DGL graph for GAT/HAS-GNN...")

    src_id = homo_graph_edges[:, 0]
    dst_id = homo_graph_edges[:, 1]

    graph = dgl.graph(
        (torch.from_numpy(src_id).long(),
         torch.from_numpy(dst_id).long()),
        num_nodes=N
    )

    graph.ndata['feat'] = torch.tensor(norm_features, dtype=torch.float32)
    graph.ndata['label'] = torch.tensor(labels_np, dtype=torch.long)
    graph.ndata['comp_id'] = comp_id
    graph.ndata['struct_score'] = struct_score
    graph.ndata['heterophily'] = heterophily
    graph.ndata['class_weight'] = class_weight

    # 训练/验证/测试 mask 这里先不划分，后面训练脚本会调用 gen_mask 动态生成
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask = torch.zeros(N, dtype=torch.bool)
    test_mask = torch.zeros(N, dtype=torch.bool)
    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    # ===================== 12. 保存到文件 =====================
    save_graphs('./data/BUPT_tele_me_chat.bin', graph,
                {'labels': graph.ndata['label']})
    save_info('./data/BUPT_tele.pkl', {'num_classes': 3})
    print('The BUPT dataset with HAS-GNN static features is successfully generated!')


if __name__ == "__main__":
    BUPT_process()
