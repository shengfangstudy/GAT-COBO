import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
import os

class BUPTFraudDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='bupt_fraud')

    def normalize(self, mx):
        """行归一化: D^-1 * A"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = spp.diags(r_inv)
        return r_mat_inv.dot(mx)

    def process(self):
        # =====================================================================
        # 1. 读取原始文件 & ID 映射
        # =====================================================================
        print("Loading BUPT raw files...")
        # 假设数据在 ./data/BUPT/ 目录下
        data_dir = "./data/BUPT/"
        feature_raw = np.genfromtxt(os.path.join(data_dir, "TF.features"), dtype=str, delimiter=' ')
        labels_raw = np.genfromtxt(os.path.join(data_dir, "TF.labels"), dtype=str, delimiter=' ')
        edges_raw = np.genfromtxt(os.path.join(data_dir, "TF.edgelist"), dtype=str, delimiter=' ')

        # --- A. 节点对齐 ---
        # 以 feature 文件中的顺序为准建立索引 0 ~ N-1
        feat_ids = feature_raw[:, 0]
        id2idx = {id_str: i for i, id_str in enumerate(feat_ids)}
        num_nodes = len(feat_ids)
        print(f"Total Nodes: {num_nodes}")

        # --- B. 特征矩阵 ---
        features = feature_raw[:, 1:].astype(np.float32)
        # Z-Score 标准化
        features = StandardScaler().fit_transform(features)
        node_features = torch.from_numpy(features)

        # --- C. 标签向量 ---
        # 建立 ID -> Label 字典
        label_dict = {row[0]: int(row[1]) for row in labels_raw}
        # 按 feat_ids 顺序生成标签数组
        labels_list = []
        for nid in feat_ids:
            if nid in label_dict:
                labels_list.append(label_dict[nid])
            else:
                # 异常处理：如果特征里有但标签里没有，默认设为 Normal(0) 或报错
                labels_list.append(0) 
        
        labels_np = np.array(labels_list, dtype=np.int64)
        node_labels = torch.from_numpy(labels_np)

        # --- D. 构建原始邻接矩阵 (用于计算指标) ---
        src_list = []
        dst_list = []
        for u, v in edges_raw:
            if u in id2idx and v in id2idx:
                src_list.append(id2idx[u])
                dst_list.append(id2idx[v])
        
        # 构造 COO 矩阵
        data = np.ones(len(src_list))
        adj_coo = spp.coo_matrix((data, (src_list, dst_list)), shape=(num_nodes, num_nodes))
        
        # 确保是对称无向图 (A + A.T)
        adj_raw = adj_coo + adj_coo.T
        # 二值化：防止多重边导致权重 > 1
        adj_raw.data = np.ones_like(adj_raw.data) 
        adj_raw = adj_raw.tocoo()

        print(f"Graph Edges (Undirected): {adj_raw.nnz // 2}")

        # =====================================================================
        # 2. HAS-GNN 静态指标预计算
        # =====================================================================
        print("Calculating HAS-GNN Static Indicators...")

        # --- Metric A: 连通子图 (comp_id) ---
        # 使用 Scipy 计算强连通分量 (无向图即弱连通)
        n_comp, comp_labels = spp.csgraph.connected_components(
            adj_raw, directed=False, return_labels=True
        )
        comp_id = torch.from_numpy(comp_labels).long()
        print(f"  - Found {n_comp} connected components.")

        # --- Metric B: 结构重要性 (struct_score) ---
        # 1. 转 NetworkX 计算全图 K-Core
        G_nx = nx.from_scipy_sparse_array(adj_raw)
        G_nx.remove_edges_from(nx.selfloop_edges(G_nx)) # 移除自环
        core_dict = nx.core_number(G_nx)
        k_core_vals = np.array([core_dict[i] for i in range(num_nodes)])

        # 2. Pandas GroupBy 计算相对分数
        df = pd.DataFrame({'comp': comp_labels, 'k': k_core_vals})
        df['max_k'] = df.groupby('comp')['k'].transform('max')
        df['max_k'] = df['max_k'].replace(0, 1) # 防止除零
        df['score'] = df['k'] / df['max_k']

        # 3. 赋值与掩码 (Masking)
        # 规则：仅对 Fraud(1) 赋予结构分，Normal(0) 和 Courier(2) 强制为 0
        struct_score_np = np.zeros(num_nodes, dtype=np.float32)
        fraud_mask_np = (labels_np == 1)
        struct_score_np[fraud_mask_np] = df.loc[fraud_mask_np, 'score'].values
        
        struct_score = torch.from_numpy(struct_score_np)
        print("  - Structural Importance calculated (Masked for Fraud only).")

        # --- Metric C: 静态异质性 (heterophily) ---
        # 定义：异类比例。
        # Fraud(1) 的异类 -> {0, 2}
        # Non-Fraud(0, 2) 的异类 -> {1}
        
        # 技巧：使用矩阵乘法计算 "欺诈邻居数量"
        # Is_Fraud 向量 (0/1)
        is_fraud_vec = (labels_np == 1).astype(float)
        # Adj * Is_Fraud = 每个节点的欺诈邻居数
        num_fraud_neighbors = adj_raw.dot(is_fraud_vec)
        
        degrees = np.array(adj_raw.sum(1)).flatten()
        degrees[degrees == 0] = 1.0 # 防止除零
        
        hetero_np = np.zeros(num_nodes, dtype=np.float32)
        
        # Case 1: 我是 Fraud(1) -> 异质性 = (总度数 - 欺诈邻居数) / 总度数
        hetero_np[fraud_mask_np] = (degrees[fraud_mask_np] - num_fraud_neighbors[fraud_mask_np]) / degrees[fraud_mask_np]
        
        # Case 2: 我是 0 或 2 -> 异质性 = 欺诈邻居数 / 总度数
        non_fraud_mask = ~fraud_mask_np
        hetero_np[non_fraud_mask] = num_fraud_neighbors[non_fraud_mask] / degrees[non_fraud_mask]
        
        heterophily = torch.from_numpy(hetero_np)
        print("  - Heterophily calculated.")

        # --- Metric D: 类别不平衡权重 (class_weight) ---
        # 规则：三类独立计算，w = log(1 + Total / (3 * Count))
        print("  - Calculating Class Weights (3-Class Independent)...")
        
        unique, counts = np.unique(labels_np, return_counts=True)
        count_dict = dict(zip(unique, counts))
        n_total = float(num_nodes)
        n_classes = 3.0
        
        class_weight_np = np.ones(num_nodes, dtype=np.float32)
        
        for c in [0, 1, 2]:
            n_c = count_dict.get(c, 0)
            n_c = max(n_c, 1) # 防止除零
            
            # 严格套用 Log 平滑公式
            w_val = np.log(1 + n_total / (n_classes * n_c))
            
            # 赋值
            class_weight_np[labels_np == c] = w_val
            
            # Log
            role = {0:"Normal", 1:"Fraud", 2:"Courier"}.get(c)
            print(f"    Class {c} ({role}): Count={n_c}, Weight={w_val:.4f}")
            
        class_weight = torch.from_numpy(class_weight_np)

        # =====================================================================
        # 3. 构建最终用于训练的 DGL 图
        # =====================================================================
        print("Building final DGL graph...")
        
        # GAT 预处理标准步骤：
        # 1. 对称化 (adj_raw 已经是了)
        # 2. 加自环 (Eye)
        # 3. 归一化 (Row Normalize)
        adj_final = adj_raw + adj_raw.T.multiply(adj_raw.T > adj_raw) - adj_raw.multiply(adj_raw.T > adj_raw)
        adj_final = self.normalize(adj_final + spp.eye(num_nodes))

        self.graph = dgl.from_scipy(adj_final)
        
        # 装填数据
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        # HAS-GNN 数据
        self.graph.ndata['comp_id'] = comp_id
        self.graph.ndata['struct_score'] = struct_score
        self.graph.ndata['heterophily'] = heterophily
        self.graph.ndata['class_weight'] = class_weight

        # =====================================================================
        # 4. 数据集划分 (Train/Val/Test)
        # =====================================================================
        # 使用 stratify 保证三类样本在各集中的比例一致
        idx_all = np.arange(num_nodes)
        
        # Train: 60%
        idx_train, idx_rest, _, y_rest = train_test_split(
            idx_all, labels_np, stratify=labels_np, train_size=0.6, random_state=42
        )
        # Val: 20% (of total), Test: 20% (of total) -> Rest split 50/50
        idx_val, idx_test, _, _ = train_test_split(
            idx_rest, y_rest, stratify=y_rest, train_size=0.5, random_state=42
        )

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[idx_train] = True
        val_mask[idx_val] = True
        test_mask[idx_test] = True

        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

        self._num_classes = 3

        # 保存
        save_graphs('./data/BUPT_tele_me.bin', self.graph, {'labels': node_labels})
        save_info('./data/BUPT_tele.pkl', {'num_classes': 3})
        print("BUPT Dataset generated successfully!")

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes

if __name__ == "__main__":
    dataset = BUPTFraudDataset()
    dataset.process()