# build_graph.py

import os
import pandas as pd
import numpy as np
import scipy.sparse as spp
from collections import defaultdict, Counter
import argparse

# ==========================================
# 1. 核心逻辑函数 (保持你的逻辑不变)
# ==========================================

def extract_contacts(voc, valid_users_set):
    """
    构建每个标记用户的直接联系人集合。
    只保留涉及 valid_users_set (即特征文件中的用户) 的通话。
    """
    print("Step 1: Extracting contacts...")
    contacts = defaultdict(set)
    
    # 优化：只遍历涉及我们关注用户的行
    # 注意：这里假设 voc 很大，但我们只关心 valid_users_set 里的用户
    # 如果 voc 非常大，建议先 filter，或者像下面这样在循环里判断
    
    for row in voc.itertuples():
        p1 = str(row.phone_no_m)
        p2 = str(row.opposite_no_m)
        call_type = row.calltype_id
        
        # 逻辑：主叫是我们关注的用户
        if call_type == 1 and p1 in valid_users_set:
            contacts[p1].add(p2)
            
        # 逻辑：被叫是我们关注的用户
        elif call_type == 2 and p2 in valid_users_set:
            contacts[p2].add(p1)
            
    return contacts

def precompute_shared_counts(contacts):
    """
    预计算所有用户对的共同联系人数量。
    """
    print("Step 2: Pre-computing shared contact counts (Inverted Index)...")
    
    # 1. 建立倒排索引：联系人 -> [用户列表]
    contact_to_users = defaultdict(list)
    for user, user_contacts in contacts.items():
        for c in user_contacts:
            contact_to_users[c].append(user)
            
    # 2. 统计 Pair 出现次数
    pair_counts = Counter()
    
    print(f"Inverted index built. Calculating pairs from {len(contact_to_users)} intermediate contacts...")
    
    for c, users in contact_to_users.items():
        if len(users) > 1:
            users.sort() # 排序确保 (A,B) 和 (B,A) 算作同一个 key
            # 只有当 users 列表长度不长时，双重循环才快。
            # 在电信数据中，一个联系人连接的用户通常不会太多，如果特别多（如热门客服），可能需要剪枝
            if len(users) > 500: continue # 剪枝策略：忽略超级热点（比如10086），避免构成完全图

            for i in range(len(users)):
                for j in range(i + 1, len(users)):
                    pair_counts[(users[i], users[j])] += 1
                    
    return pair_counts

def build_adjacency_matrix(pair_counts, k, user_to_idx, num_nodes):
    """
    构建稀疏邻接矩阵。
    """
    print(f"Step 3: Building adjacency matrix for K={k}...")
    
    row = []
    col = []
    data = []
    
    edge_count = 0
    for (u1, u2), count in pair_counts.items():
        if count >= k:
            # 必须确保 u1 和 u2 都在我们的特征用户列表中
            if u1 in user_to_idx and u2 in user_to_idx:
                idx1 = user_to_idx[u1]
                idx2 = user_to_idx[u2]
                
                # 添加边 (无向图，通常添加双向，或者只添加单向由DGL处理)
                # DGLDataset代码里有 adj + adj.T 操作，但为了保险，我们在这里构建完整的对称矩阵
                # u1 -> u2
                row.append(idx1)
                col.append(idx2)
                data.append(1)
                
                # u2 -> u1
                row.append(idx2)
                col.append(idx1)
                data.append(1)
                
                edge_count += 1
    
    print(f"Total edges (bidirectional) constructed: {len(row)}")
    
    # 构建 COO 矩阵
    # shape 必须是 (num_nodes, num_nodes)
    adj = spp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes), dtype=np.int8)
    
    return adj

# ==========================================
# 2. 主执行流程
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=1, help='Minimum shared contacts to form an edge')
    parser.add_argument('--data_path', type=str, default='./data/Sichuan/', help='Path to data files')
    args = parser.parse_args()

    # 路径配置
    feat_path = os.path.join(args.data_path, "all_feat_with_label.csv")
    voc_path = os.path.join(args.data_path, "voc.csv")
    output_path = os.path.join(args.data_path, "node_adj_sparse_zhu.npz")

    # 1. 读取特征文件以获取节点顺序 (Crucial Step!)
    print(f"Loading feature file from {feat_path} to establish node order...")
    # 读取所有列可能太慢，只读取第一列 phone_no_m
    # 注意：all_feat_with_label.csv 必须包含 header，且第一列是 phone_no_m
    try:
        # 使用 dtype=str 防止手机号被读成数字
        df_feat = pd.read_csv(feat_path, usecols=[0], dtype=str) 
        # 假设第一列的名字不一定是 'phone_no_m'，我们直接取第一列
        node_list = df_feat.iloc[:, 0].tolist()
    except Exception as e:
        print(f"Error reading feature file: {e}")
        exit()

    num_nodes = len(node_list)
    # 构建映射: phone_number -> matrix_index
    user_to_idx = {u: i for i, u in enumerate(node_list)}
    valid_users_set = set(node_list)
    
    print(f"Total nodes: {num_nodes}")

    # 2. 读取通话记录 (VOC)
    print(f"Loading VOC file from {voc_path}...")
    try:
        voc = pd.read_csv(voc_path, dtype={'phone_no_m': str, 'opposite_no_m': str})
    except FileNotFoundError:
        print("Error: voc.csv not found.")
        exit()

    # 3. 提取联系人
    contacts = extract_contacts(voc, valid_users_set)

    # 4. 计算共同联系人
    pair_counts = precompute_shared_counts(contacts)
    print(f"Total user pairs with shared contacts: {len(pair_counts)}")

    # 5. 构建并保存邻接矩阵
    adj_matrix = build_adjacency_matrix(pair_counts, args.k, user_to_idx, num_nodes)
    
    print(f"Saving adjacency matrix to {output_path}...")
    spp.save_npz(output_path, adj_matrix)
    
    print("Done! The graph is ready for DGL.")