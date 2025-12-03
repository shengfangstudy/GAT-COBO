import matplotlib
# === 关键优化 1：强制使用非交互式后端 ===
matplotlib.use('Agg') 

import logging
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import dgl
import os
import sys
import datetime
import json
import time 
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import f1_score, classification_report, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from dgl.data.utils import load_graphs, load_info
from imblearn.metrics import geometric_mean_score

# 导入自定义模块
from utils import EarlyStopping, misclassification_cost, _set_cost_matrix, cost_table_calc, _validate_cost_matrix
# [关键] 导入 BUPT 专用的动态因子计算 (G/D 仅对 Fraud 生效)
from utilsme2 import calc_dynamic_factors 
from model import GAT_COBO

# ====================== 0. 实验记录与可视化工具类 ======================
class ExperimentManager:
    def __init__(self, base_dir="training_records_bupt"):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.run_dir)
        
        self.global_log_path = os.path.join(base_dir, "all_training_logs.log")
        self.metrics = {} 
        self.start_time = time.time()
        self._setup_plot_fonts()
        
        print(f"Experimental results will be saved to: {self.run_dir}")

    def _setup_plot_fonts(self):
        plt.rcParams['axes.unicode_minus'] = False
        try:
            import matplotlib.font_manager as fm
            system_fonts = set(f.name for f in fm.fontManager.ttflist)
            chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
            for font in chinese_fonts:
                if font in system_fonts:
                    plt.rcParams['font.sans-serif'] = [font]
                    break
        except:
            pass

    def setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.handlers = []
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        file_handler = logging.FileHandler(self.global_log_path, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        logging.info(f"\n{'='*20} New Experiment Started: {self.timestamp} {'='*20}")
        logging.info(f"Run Directory: {self.run_dir}")

    def record_epoch(self, layer_idx, epoch, metrics_dict):
        if layer_idx not in self.metrics:
            self.metrics[layer_idx] = {k: [] for k in metrics_dict.keys()}
            self.metrics[layer_idx]['epochs'] = []

        self.metrics[layer_idx]['epochs'].append(epoch)
        for k, v in metrics_dict.items():
            self.metrics[layer_idx][k].append(v)

    def render_plot(self, layer_idx):
        if layer_idx not in self.metrics:
            return

        epochs = self.metrics[layer_idx]['epochs']
        data = self.metrics[layer_idx]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Layer {layer_idx+1} Training Progress", fontsize=16)

        metrics_map = {
            (0,0): ('Loss', 'train_loss', 'val_loss', 'r'),
            (0,1): ('AUC', 'train_auc', 'val_auc', 'b'),
            (1,0): ('F1', 'train_f1', 'val_f1', 'g'),
            (1,1): ('Recall', 'train_recall', 'val_recall', 'm')
        }
        
        for (r, c), (title, tk, vk, color) in metrics_map.items():
            ax = axes[r, c]
            ax.plot(epochs, data[tk], f'{color}-', label=f'Train {title}')
            ax.plot(epochs, data[vk], f'{color}--', label=f'Val {title}')
            ax.set_title(f'{title} Curve')
            ax.set_xlabel('Epoch')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        final_path = os.path.join(self.run_dir, f"layer_{layer_idx+1}_metrics.png")
        try:
            plt.savefig(final_path)
        finally:
            plt.close(fig)

    def save_experiment_summary(self, args, final_metrics):
        duration_sec = time.time() - self.start_time
        m, s = divmod(duration_sec, 60)
        
        summary = {
            "timestamp": self.timestamp,
            "duration": f"{int(m)}m {int(s)}s",
            "args": vars(args),
            "metrics": final_metrics
        }
        
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
            
        with open(os.path.join(self.run_dir, "summary.json"), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, default=convert)

# ====================== 1. 基础配置 ======================
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    dgl.random.seed(seed)

def get_args():
    parser = argparse.ArgumentParser(description='HAS-GNN Boosting BUPT')
    # === Dataset & Model ===
    parser.add_argument('--dataset', type=str, default='BUPT')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--hid', type=int, default=64) 
    
    # Dropout
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--adj_dropout", type=float, default=0.1)
    parser.add_argument("--attn_drop", type=float, default=0.1)
    parser.add_argument("--in_drop", type=float, default=0.1)
    
    # Boosting Coefficients
    parser.add_argument('--attention_weight', type=float, default=0.5)
    parser.add_argument('--feature_weight', type=float, default=0.6)
    
    # Structure
    parser.add_argument('--layers', type=int, default=2, help='Number of Boosting Layers')
    parser.add_argument("--num_layers", type=int, default=1, help="GAT layers inside each booster")
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument("--num_out_heads", type=int, default=1)
    
    # Training
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--att_loss_weight', type=float, default=0.01) # BUPT 常用 0.01
    parser.add_argument('--print_interval', type=int, default=50)
    
    # Imbalance & Cost
    parser.add_argument('--IR', type=float, default=0.1)
    parser.add_argument('--IR_set', type=int, default=0)
    parser.add_argument('--cost', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--blank', type=int, default=0)

    # === HAS-GNN Hyperparameters (建议 0.5 防止爆炸) ===
    parser.add_argument('--lambda_I', type=float, default=1.0, help='Coef for Structural Importance')
    parser.add_argument('--lambda_G', type=float, default=1.0, help='Coef for Group Difficulty')
    parser.add_argument('--lambda_D', type=float, default=1.0, help='Coef for Camouflage Difficulty')

    return parser.parse_args()

# ====================== 2. 指标计算与掩码生成 ======================
def calculate_metrics(logits, labels, n_classes):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    acc = (preds == labels).sum().item() / len(labels)
    # 3分类使用 Macro Average
    f1 = f1_score(labels_np, preds_np, average='macro')
    recall = recall_score(labels_np, preds_np, average='macro')
    gmean = geometric_mean_score(labels_np, preds_np, average='macro')
    try:
        # 多分类 AUC (One-vs-One)
        auc = roc_auc_score(labels_np, probs_np, average='macro', multi_class='ovo')
    except:
        auc = 0.0 
    return acc, f1, recall, gmean, auc

def load_data(args, device):
    # 加载 HAS-GNN 处理后的 BUPT 数据
    dataset, _ = load_graphs("./data/BUPT_tele_me.bin")
    n_classes = 3
    g = dataset[0]
    g = g.int().to(device)
    for e in g.etypes:
        g = dgl.remove_self_loop(g, etype=e) 
        g = dgl.add_self_loop(g, etype=e) 
    return g, g.ndata['feat'].float(), g.ndata['label'].long(), n_classes

# ====================== 3. 训练逻辑 (Boosting Weak Learner) ======================
def train_single_layer(layer_idx, model, optimizer, g, features, labels, sample_weights, n_classes, args, stopper, exp_manager):
    """
    使用固定的 sample_weights 进行加权训练
    """
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']
    
    if args.early_stop:
        stopper.best_score = None
        stopper.counter = 0
        stopper.early_stop = False

    for epoch in range(args.epochs):
        model.train()
        logits, logits_GAT, _ = model(features)
        
        logits = logits.view(logits.shape[0], -1)
        logits_GAT = logits_GAT.view(logits_GAT.shape[0], -1)

        # 加权 Loss (Sample Weights 由 Boosting/HAS-GNN 决定)
        ce_node = F.cross_entropy(logits[train_mask], labels[train_mask], reduction='none')
        loss_node = (ce_node * sample_weights).sum()

        ce_att = F.cross_entropy(logits_GAT[train_mask], labels[train_mask], reduction='none')
        loss_att = args.att_loss_weight * (ce_att * sample_weights).sum()

        loss = loss_node + loss_att

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 验证
        with torch.no_grad():
            t_acc, t_f1, t_rec, t_gmean, t_auc = calculate_metrics(logits[train_mask], labels[train_mask], n_classes)
            
            model.eval()
            v_logits, _, _ = model(features)
            v_logits = v_logits.view(v_logits.shape[0], -1)
            v_loss = F.cross_entropy(v_logits[val_mask], labels[val_mask]).item()
            v_acc, v_f1, v_rec, v_gmean, v_auc = calculate_metrics(v_logits[val_mask], labels[val_mask], n_classes)
            model.train() 

        metrics_dict = {
            'train_loss': loss.item(), 'val_loss': v_loss,
            'train_auc': t_auc,        'val_auc': v_auc,
            'train_f1': t_f1,          'val_f1': v_f1,
            'train_recall': t_rec,     'val_recall': v_rec
        }
        exp_manager.record_epoch(layer_idx, epoch, metrics_dict)

        if epoch % args.print_interval == 0:
            exp_manager.render_plot(layer_idx)
            logging.info(f"[Layer {layer_idx+1}] Ep {epoch}: Loss={loss.item():.4f}/{v_loss:.4f} | AUC={t_auc:.4f}/{v_auc:.4f}")
        
        if args.early_stop and stopper.step(v_loss, model, epoch):
            logging.info(f"Early stopping triggered at epoch {epoch}")
            exp_manager.render_plot(layer_idx)
            break
    
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    return model

def update_boosting(model, features, labels, train_mask, sample_weights, cost_matrix, n_classes, g, args, device):
    """
    [修正] 回归 Sichuan 原版代码的 exp 更新逻辑，并注入 HAS-GNN 因子
    """
    model.eval()
    with torch.no_grad():
        ada_use_h, _, attention = model(features)

    # 1. 计算 H (Eq.10, 12 in GAT-COBO paper)
    output_logp = torch.log(F.softmax(ada_use_h, dim=1))
    h = (n_classes - 1) * (output_logp - torch.mean(output_logp, dim=1).view(-1, 1))
    
    # 2. 计算 Cost (Eq.13)
    y_pred = torch.argmax(h[train_mask], dim=1)
    base_cost = torch.tensor(misclassification_cost(labels[train_mask].cpu(), y_pred.cpu(), cost_matrix), device=device)
    
    # ============================================================
    # [HAS-GNN 注入] 
    # 原逻辑: exp(alpha * cost)
    # 新逻辑: exp(alpha * cost * HAS_Factor)
    # HAS_Factor = (1+I)(1+G)(1+D)
    # ============================================================
    
    # A. 静态结构 I
    I_val = g.ndata['struct_score'][train_mask].to(device)
    
    # B. 动态因子 G, D (utilsme_bupt 已处理好 0/2 类屏蔽逻辑)
    full_G, full_D = calc_dynamic_factors(ada_use_h, g, args, device)
    G_val = full_G[train_mask]
    D_val = full_D[train_mask]
    
    # C. 计算 HAS 放大倍数 (Modifier)
    # 我们加上 clamp 保护，防止倍数过大导致 exp 溢出
    has_modifier = (1.0 + args.lambda_I * I_val) * \
                   (1.0 + args.lambda_G * G_val) * \
                   (1.0 + args.lambda_D * D_val)
    has_modifier = torch.clamp(has_modifier, max=5.0) 
    
    # ============================================================
    
    # 3. 计算 Estimator Weight (Alpha)
    temp = F.nll_loss(F.log_softmax(ada_use_h[train_mask], 1), labels[train_mask], reduction='none')
    estimator_weight = (n_classes - 1) / n_classes * temp
    # 限制 alpha 大小，防止过拟合时权重波动太大
    estimator_weight = torch.clamp(estimator_weight, max=2.0) 
    
    # 4. 更新权重 (严格对应四川代码 Eq.14)
    # 原版: weight = sample_weights * torch.exp(estimator_weight * cost * mask)
    # 修改版: 将 has_modifier 乘在 cost 上，作为"结构化代价"
    
    exponent = estimator_weight * base_cost * has_modifier
    exponent = torch.clamp(exponent, max=5.0) # 再次保护 exp 指数
    
    # 这里的 mask 逻辑完全保留原版: ((sample_weights > 0) | (estimator_weight < 0))
    # 只有当样本权重>0 或者 alpha<0 时才更新
    mask = ((sample_weights > 0) | (estimator_weight < 0)).float()
    
    new_weights = sample_weights * torch.exp(exponent * mask)
    
    # 5. 归一化
    sample_weights = (new_weights / new_weights.sum()).detach()

    # (Feature 更新保持不变)
    attention = attention.view(attention.shape[0], -1)
    row, col = g.edges()[0].cpu().numpy(), g.edges()[1].cpu().numpy()
    att_data = attention.cpu().numpy().T.squeeze()
    
    adj_att = torch.sparse_coo_tensor(torch.tensor([row, col]).to(device), torch.tensor(att_data).to(device), (g.num_nodes(), g.num_nodes()))
    new_features = torch.sparse.mm(args.attention_weight * adj_att, args.feature_weight * features).detach()
    
    return h, sample_weights, new_features

# ====================== 4. 主流程 ======================
def main():
    args = get_args()
    setup_seed(args.seed)
    
    exp_manager = ExperimentManager()
    exp_manager.setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Arguments: {args}")

    g, features, labels, n_classes = load_data(args, device)
    train_mask, test_mask = g.ndata['train_mask'], g.ndata['test_mask']

    final_results = torch.zeros(g.num_nodes(), n_classes).to(device)
    
    # ============================================================
    # Layer 0 初始化：静态先验 (Class + Struct)
    # ============================================================
    logging.info("Initializing weights using Static HAS-GNN Priors...")
    
    w_class = g.ndata['class_weight'][train_mask].to(device)
    w_struct = 1.0 + args.lambda_I * g.ndata['struct_score'][train_mask].to(device)
    
    init_weights = w_class * w_struct
    sample_weights = init_weights / init_weights.sum()

    how_dic = {0:'uniform', 1:'inverse', 2:'log1p-inverse'}
    cost_matrix_df = cost_table_calc(_validate_cost_matrix(_set_cost_matrix(labels[train_mask].cpu(), how=how_dic[args.cost]), n_classes))
    stopper = EarlyStopping(args.patience) if args.early_stop else None

    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT_COBO(g, args.num_layers, features.shape[1], args.hid, n_classes, heads, F.elu, 
                        args.dropout, args.adj_dropout, args.in_drop, args.attn_drop, args.negative_slope, args.residual).to(device)

    for layer in range(args.layers):
        if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
        logging.info(f"|This is the {layer + 1}th layer!|")

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train
        model = train_single_layer(layer, model, optimizer, g, features, labels, sample_weights, n_classes, args, stopper, exp_manager)
        
        # Boosting Update (使用恢复了 exp 逻辑的函数)
        h, sample_weights, features = update_boosting(model, features, labels, train_mask, sample_weights, cost_matrix_df, n_classes, g, args, device)
        final_results += h

    # Final Evaluation
    test_logits = final_results[test_mask]
    if torch.isnan(test_logits).any(): test_logits = torch.nan_to_num(test_logits)
    
    t_acc, t_f1, t_rec, t_gmean, t_auc = calculate_metrics(test_logits, labels[test_mask], n_classes)

    preds = torch.argmax(test_logits, dim=1).cpu().numpy()
    labels_np = labels[test_mask].cpu().numpy()
    target_names = ['Normal', 'Fraud', 'Courier']
    report = classification_report(labels_np, preds, target_names=target_names, digits=4)
    logging.info(f"\nFinal Test Report:\n{report}")
    
    result_str = (f"Final Result -> Macro AUC: {t_auc:.4f}, Macro F1: {t_f1:.4f}, Macro Recall: {t_rec:.4f}, G-mean: {t_gmean:.4f}, Acc: {t_acc:.4f}")
    logging.info(result_str)
    
    final_metrics = {"Macro AUC": t_auc, "Macro F1": t_f1, "Macro Recall": t_rec, "G-mean": t_gmean, "Accuracy": t_acc}
    exp_manager.save_experiment_summary(args, final_metrics)

if __name__ == "__main__":
    main()