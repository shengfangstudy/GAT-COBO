import dgl
from dgl.data import DGLDataset
import torch
import numpy as np
import scipy.sparse as spp
from sklearn.preprocessing import StandardScaler
from dgl.data.utils import save_graphs, save_info


class TelcomFraudDataset(DGLDataset):
    def __init__(self):
        super().__init__(name='telcom_fraud')

    def normalize(self,mx):
        # Row-normalize sparse matrix
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = spp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def process(self,path="./data/Sichuan/"):
        # load raw feature and labels
        idx_features_labels = np.genfromtxt("{}{}.csv".format(path, "all_feat_with_label"),
                                            dtype=np.dtype(str), delimiter=',', skip_header=1)
        # 去掉第 0 列的 phone_no_m 和最后一列的 label，只取特征，用 csr_matrix 转为稀疏矩阵格式，方便后续处理/节省内存
        features = spp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)

        #normalize the feature with z-score
        # features=StandardScaler().fit_transform(features.todense())
        # 让每个特征维度均值为 0、方差为 1
        # features.todense()：从 CSR 转成 dense 矩阵
        features = StandardScaler().fit_transform(np.asarray(features.todense()))

        # 取得标签
        labels = np.array(idx_features_labels[:, -1], dtype=np.int_)
        self.labels=torch.tensor(labels)
        node_features = torch.from_numpy(np.array(features))
        node_labels = torch.from_numpy(labels)

        # load adjacency matrix
        #　从 node_adj_sparse.npz 读取邻接矩阵（稀疏）
        adj = spp.load_npz(path + 'node_adj_sparse_zhu.npz')
        # .toarray() 先变成 dense，然后再 coo_matrix,这里有一点点绕，本质上是确保类型为 coo_matrix，方便后续运算
        adj = adj.toarray()
        adj = spp.coo_matrix(adj)

        # build symmetric adjacency matrix and normalize
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = self.normalize(adj + spp.eye(adj.shape[0]))

        self.graph = dgl.from_scipy(adj)
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels

        # specify the default train,valid,test set for DGLgraph
        n_nodes = features.shape[0]
        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)
        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.num_labels = 2
        self._num_classes=2

        save_graphs('./data/Sichuan_tele_zhu.bin', self.graph, {'labels': self.labels})
        save_info('./data/Sichuan_tele.pkl', {'num_classes': self.num_classes})
        print('The Sichuan dataset is successfully generated! ')

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