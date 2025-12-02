import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info
import networkx as nx


class TelcomFraudDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='telcom_fraud')

    def normalize(self, mx):
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = spp.diags(r_inv)
        return r_mat_inv.dot(mx)

    def process(self, path="./data/Sichuan/"):

        print("Loading raw features and labels...")
        # ====================== A. 加载特征和标签 ========================
        idx_features_labels = np.genfromtxt(
            f"{path}all_feat_with_label.csv",
            dtype=np.dtype(str), delimiter=',', skip_header=1
        )

        # 特征
        features = spp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
        features = StandardScaler().fit_transform(features.A)  # A=toarray()

        # 标签
        labels_np = np.array(idx_features_labels[:, -1], dtype=np.int_)
        labels = torch.from_numpy(labels_np)

        node_features = torch.from_numpy(features)
        node_labels = labels.clone()

        # ====================== B. 加载“原始二值图” ========================
        print("Loading raw binary adjacency...")
        adj_raw = spp.load_npz(path + 'node_adj_sparse_zhu.npz').tocoo()

        n_nodes = adj_raw.shape[0]

        # ---------------------------------------------------------------------
        # 正确：所有图结构计算（comp_id、k-core、heterophily）都基于 adj_raw
        # ---------------------------------------------------------------------

        # ====================== C. 连通子图（comp_id） ======================
        print("Computing connected components...")

        # 使用 sklearn/scipy 实现，速度最快
        n_comp, comp_labels = spp.csgraph.connected_components(
            adj_raw, directed=False, return_labels=True
        )
        comp_id = torch.from_numpy(comp_labels).long()

        # ====================== D. 结构重要性（k-core） ======================
        print("Computing structural importance (k-core)...")

        # 将 raw adjacency 转为 NX 无向图
        G_nx = nx.from_scipy_sparse_array(adj_raw)  # 100% 使用原始图
        G_nx.remove_edges_from(nx.selfloop_edges(G_nx))

        struct_score = torch.zeros(n_nodes, dtype=torch.float32)

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
                if labels_np[v] == 1:  # 仅对 fraud 生效
                    struct_score[v] = core_dict[v] / cmax

        # ====================== E. 静态异质性 heterophily ====================
        print("Computing static heterophily...")

        # Fraud 邻居个数 = A * labels
        num_fraud_neighbors = adj_raw.dot(labels_np)

        degree = np.array(adj_raw.sum(1)).flatten()
        degree[degree == 0] = 1

        hetero = np.zeros(n_nodes, dtype=np.float32)

        fraud_mask = (labels_np == 1)
        normal_mask = (labels_np == 0)

        # Fraud 节点：normal邻居比例 = (deg - fraud_nb) / deg
        hetero[fraud_mask] = (degree[fraud_mask] - num_fraud_neighbors[fraud_mask]) / degree[fraud_mask]

        # Normal 节点：fraud邻居比例 = fraud_nb / deg
        hetero[normal_mask] = num_fraud_neighbors[normal_mask] / degree[normal_mask]

        heterophily = torch.from_numpy(hetero)

        # ====================== F. 类别不平衡权重（Log 平滑） ============================
        print("Computing class weights (log-smoothed)...")

        n_pos = np.sum(labels_np == 1)
        n_neg = np.sum(labels_np == 0)
        N = n_pos + n_neg

        # Inverse-frequency 的 log 平滑版本
        w_pos = np.log(1 + N / (2 * max(n_pos, 1)))
        w_neg = np.log(1 + N / (2 * max(n_neg, 1)))

        class_weight = torch.where(
            labels == 1,
            torch.tensor(w_pos, dtype=torch.float32),
            torch.tensor(w_neg, dtype=torch.float32)
        )

        # ====================== G. 构建最终 GAT 图 ===========================
        print("Building final DGL graph for GAT...")

        # 只用于 GNN 的邻接矩阵：对称化 + 加自环 + normalize
        adj = adj_raw + adj_raw.T.multiply(adj_raw.T > adj_raw) - adj_raw.multiply(adj_raw.T > adj_raw)
        adj = self.normalize(adj + spp.eye(n_nodes))

        g = dgl.from_scipy(adj)

        # Add node data
        g.ndata['feat'] = node_features
        g.ndata['label'] = node_labels
        g.ndata['comp_id'] = comp_id
        g.ndata['struct_score'] = struct_score
        g.ndata['heterophily'] = heterophily
        g.ndata['class_weight'] = class_weight

        # ====================== H. Mask ======================================
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)

        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True

        g.ndata['train_mask'] = train_mask
        g.ndata['val_mask'] = val_mask
        g.ndata['test_mask'] = test_mask

        self.graph = g
        self._num_classes = 2

        save_graphs('./data/Sichuan_tele_zhu.bin', g, {'labels': labels})
        save_info('./data/Sichuan_tele.pkl', {'num_classes': self.num_classes})

        print("Dataset generated successfully!")

    def __getitem__(self, i):
        return self.graph

    def __len__(self):
        return 1

    @property
    def num_classes(self):
        return self._num_classes
    
if __name__=="__main__":
    # process Sichuan dataset
    dataset = TelcomFraudDataset()
