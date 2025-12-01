import matplotlib
# === 关键优化 1：强制使用非交互式后端，必须在 import pyplot 之前设置 ===
# 这能极大提升服务器端的绘图速度，并避免 X11/Font 冲突
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
import time # 用于计时
import matplotlib.pyplot as plt
import shutil
from sklearn.metrics import f1_score, classification_report, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from dgl.data.utils import load_graphs, load_info
from imblearn.metrics import geometric_mean_score

# 导入自定义模块
from utils import EarlyStopping, misclassification_cost, _set_cost_matrix, cost_table_calc, _validate_cost_matrix
from model import GAT_COBO

# ====================== 0. 实验记录与可视化工具类 ======================

class ExperimentManager:
    def __init__(self, base_dir="training_records_sichuan_0"):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        
        self.timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.run_dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.run_dir)
        
        self.global_log_path = os.path.join(base_dir, "all_training_logs.log")
        
        # 数据容器
        self.metrics = {} 
        
        # 记录开始时间
        self.start_time = time.time()
        
        # 配置绘图字体
        self._setup_plot_fonts()
        
        print(f"Experimental results will be saved to: {self.run_dir}")

    def _setup_plot_fonts(self):
        """一次性配置字体，避免循环中重复搜索"""
        plt.rcParams['axes.unicode_minus'] = False
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
        try:
            import matplotlib.font_manager as fm
            system_fonts = set(f.name for f in fm.fontManager.ttflist)
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
        
        # 彻底屏蔽 Matplotlib 日志
        logging.getLogger('matplotlib').setLevel(logging.ERROR)
        
        logging.info(f"\n{'='*20} New Experiment Started: {self.timestamp} {'='*20}")
        logging.info(f"Run Directory: {self.run_dir}")

    def record_epoch(self, layer_idx, epoch, metrics_dict):
        """
        === 关键优化 2：只记录数据，不绘图 ===
        这个函数运行极快，每个 epoch 调用都不会卡顿
        """
        if layer_idx not in self.metrics:
            self.metrics[layer_idx] = {k: [] for k in metrics_dict.keys()}
            self.metrics[layer_idx]['epochs'] = []

        self.metrics[layer_idx]['epochs'].append(epoch)
        for k, v in metrics_dict.items():
            self.metrics[layer_idx][k].append(v)

    def render_plot(self, layer_idx):
        """
        === 关键优化 2：批量绘图 ===
        只在特定间隔调用此函数，生成图片文件
        """
        if layer_idx not in self.metrics:
            return

        epochs = self.metrics[layer_idx]['epochs']
        data = self.metrics[layer_idx]
        
        # 创建画布 (使用 Agg 后端，内存绘图)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Layer {layer_idx+1} Training Progress", fontsize=16)

        # 1. Loss
        ax = axes[0, 0]
        ax.plot(epochs, data['train_loss'], 'r-', label='Train Loss')
        ax.plot(epochs, data['val_loss'], 'r--', label='Val Loss')
        ax.set_title('Loss Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. AUC
        ax = axes[0, 1]
        ax.plot(epochs, data['train_auc'], 'b-', label='Train AUC')
        ax.plot(epochs, data['val_auc'], 'b--', label='Val AUC')
        ax.set_title('AUC Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('AUC')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Macro F1
        ax = axes[1, 0]
        ax.plot(epochs, data['train_f1'], 'g-', label='Train Macro F1')
        ax.plot(epochs, data['val_f1'], 'g--', label='Val Macro F1')
        ax.set_title('Macro F1 Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Macro Recall
        ax = axes[1, 1]
        ax.plot(epochs, data['train_recall'], 'm-', label='Train Macro Recall')
        ax.plot(epochs, data['val_recall'], 'm--', label='Val Macro Recall')
        ax.set_title('Macro Recall Curve')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 原子写入文件
        final_path = os.path.join(self.run_dir, f"layer_{layer_idx+1}_metrics.png")
        temp_path = final_path + ".tmp"
        
        try:
            plt.savefig(temp_path, format='png') 
        except Exception as e:
            logging.error(f"Plot saving failed: {e}")
        finally:
            plt.close(fig) # 必须关闭，释放内存
        
        try:
            os.replace(temp_path, final_path)
        except OSError:
            shutil.move(temp_path, final_path)

    def save_experiment_summary(self, args, final_metrics):
        # 计算总耗时
        end_time = time.time()
        duration_sec = end_time - self.start_time
        m, s = divmod(duration_sec, 60)
        time_str = f"{int(m)}m {int(s)}s"

        summary = {
            "timestamp": self.timestamp,
            "total_duration": time_str,
            "args": vars(args),
            "final_metrics": final_metrics
        }
        json_path = os.path.join(self.run_dir, "experiment_summary.json")
        def convert(o):
            if isinstance(o, np.generic): return o.item()
            raise TypeError
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False, default=convert)
        
        logging.info(f"Experiment finished in {time_str}")
        logging.info(f"Experiment summary saved to {json_path}")


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
    parser = argparse.ArgumentParser(description='GAT-COBO for Sichuan Dataset')
    # === 四川数据集特定参数 ===
    parser.add_argument('--dataset', type=str, default='Sichuan', help='Dataset Name')
    parser.add_argument('--train_size', type=float, default=0.6, help='Train set ratio') 
    parser.add_argument('--lr', type=float, default=0.002, help='Initial learning rate') 
    parser.add_argument('--hid', type=int, default=128, help='Hidden units') 
    
    # Dropout 相关
    parser.add_argument("--dropout", type=float, default=0, help="Feature dropout")
    parser.add_argument("--adj_dropout", type=float, default=0.4, help="Adjacency dropout")
    parser.add_argument("--attn_drop", type=float, default=0.5, help="Attention dropout")
    
    # 权重系数
    parser.add_argument('--attention_weight', type=float, default=0.1, help='Attention coefficient')
    parser.add_argument('--feature_weight', type=float, default=0.1, help='Feature adjust coefficient')
    
    # === 通用/其他参数 ===
    parser.add_argument('--layers', type=int, default=2, help='Boosting layers')
    parser.add_argument("--num_layers", type=int, default=1, help="GAT layers")
    parser.add_argument('--num_heads', type=int, default=1, help='Attention heads')
    parser.add_argument("--num_out_heads", type=int, default=1, help="Output heads")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay')
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument('--patience', type=int, default=60, help='Early stopping patience')
    parser.add_argument("--in_drop", type=float, default=0.1, help="input feature dropout")
    parser.add_argument('--early_stop', action='store_true', default=False, help="Use early stop")
    parser.add_argument("--residual", action="store_true", default=False, help="Use residual")
    parser.add_argument('--negative_slope', type=float, default=0.2, help="LeakyReLU slope")
    parser.add_argument('--att_loss_weight', type=float, default=0.5, help="attention loss weight")
    parser.add_argument('--IR', type=float, default=0.1, help='Imbalanced ratio')
    parser.add_argument('--IR_set', type=int, default=0, help='Set IR manually')
    parser.add_argument('--cost', type=int, default=2, help="Cost matrix type")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--print_interval', type=int, default=50, help="Log interval")
    parser.add_argument('--blank', type=int, default=0, help='Debug flag')

    return parser.parse_args()

# ====================== 2. 数据处理与指标计算 ======================

def calculate_metrics(logits, labels, n_classes):
    probs = torch.softmax(logits, dim=1)
    preds = torch.argmax(probs, dim=1)
    labels_np = labels.cpu().numpy()
    preds_np = preds.cpu().numpy()
    probs_np = probs.detach().cpu().numpy()

    acc = (preds == labels).sum().item() / len(labels)
    f1 = f1_score(labels_np, preds_np, average='macro')
    recall = recall_score(labels_np, preds_np, average='macro')
    gmean = geometric_mean_score(labels_np, preds_np)
    try:
        if n_classes == 2:
            auc = roc_auc_score(labels_np, probs_np[:, 1], average='macro')
        else:
            auc = roc_auc_score(labels_np, probs_np, average='macro', multi_class='ovo')
    except Exception:
        auc = 0.0 

    return acc, f1, recall, gmean, auc

def gen_mask(g, train_rate, val_rate, IR, IR_set):
    labels = g.ndata['label'].long().numpy()
    n_nodes = len(labels)
    index = list(range(n_nodes))

    if IR_set != 0:
        fraud_idx = np.where(labels == 1)[0].tolist()
        benign_idx = np.where(labels == 0)[0].tolist()
        current_IR = len(fraud_idx) / len(benign_idx)
        if IR < current_IR:
            n_sample = int(IR * len(benign_idx))
            sampled_fraud = random.sample(fraud_idx, n_sample)
            index = benign_idx + sampled_fraud
        else:
            n_sample = int(len(fraud_idx) / IR)
            sampled_benign = random.sample(benign_idx, n_sample)
            index = sampled_benign + fraud_idx

    sub_labels = labels[index]
    train_idx, val_test_idx, _, y_vt = train_test_split(index, sub_labels, stratify=sub_labels, train_size=train_rate, random_state=2, shuffle=True)
    val_idx, test_idx, _, _ = train_test_split(val_test_idx, y_vt, train_size=val_rate/(1-train_rate), random_state=2, shuffle=True)

    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g

def load_data(args, device):
    dataset, _ = load_graphs("./data/Sichuan_tele.bin")
    n_classes = load_info("./data/Sichuan_tele.pkl")['num_classes']
    g = dataset[0]
    g = gen_mask(g, args.train_size, 0.2, args.IR, args.IR_set)
    g = g.int().to(device)
    for e in g.etypes:
        g = dgl.remove_self_loop(g, etype=e) 
        g = dgl.add_self_loop(g, etype=e) 
    return g, g.ndata['feat'].float(), g.ndata['label'].long(), n_classes

# ====================== 3. 训练逻辑 ======================
def train_single_layer(layer_idx, model, optimizer, g, features, labels, sample_weights, n_classes, args, stopper, exp_manager):
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

        # 计算 Loss
        loss_node = F.nll_loss(F.log_softmax(logits[train_mask], 1), labels[train_mask], reduction='none')
        loss_att = args.att_loss_weight * F.nll_loss(F.log_softmax(logits_GAT[train_mask], 1), labels[train_mask], reduction='none')
        loss_tensor = loss_node + loss_att
        loss = (loss_tensor * sample_weights).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # === 实时计算 Train 和 Val 的所有指标 ===
        # 这些计算是基于 Tensor 操作的，比文件 IO 快得多
        with torch.no_grad():
            t_acc, t_f1, t_rec, t_gmean, t_auc = calculate_metrics(logits[train_mask], labels[train_mask], n_classes)
            
            model.eval()
            v_logits, _, _ = model(features)
            v_logits = v_logits.view(v_logits.shape[0], -1)
            v_loss = F.cross_entropy(v_logits[val_mask], labels[val_mask]).item()
            v_acc, v_f1, v_rec, v_gmean, v_auc = calculate_metrics(v_logits[val_mask], labels[val_mask], n_classes)
            model.train() 

        # 1. 内存记录（每个 epoch 都做，很快）
        metrics_dict = {
            'train_loss': loss.item(), 'val_loss': v_loss,
            'train_auc': t_auc,        'val_auc': v_auc,
            'train_f1': t_f1,          'val_f1': v_f1,
            'train_recall': t_rec,     'val_recall': v_rec
        }
        exp_manager.record_epoch(layer_idx, epoch, metrics_dict)

        # 2. 磁盘绘图（每隔 print_interval 做一次，较慢）
        if epoch % args.print_interval == 0:
            exp_manager.render_plot(layer_idx)
            
            # 打印日志
            log_msg = (f"[Layer {layer_idx+1}] Ep {epoch}: "
                       f"Loss={loss.item():.4f}/{v_loss:.4f} | "
                       f"AUC={t_auc:.4f}/{v_auc:.4f} | "
                       f"F1={t_f1:.4f}/{v_f1:.4f} | "
                       f"Rec={t_rec:.4f}/{v_rec:.4f} | "
                       f"G-Mean={t_gmean:.4f}/{v_gmean:.4f}")
            logging.info(log_msg)
        
        if args.early_stop and stopper.step(v_loss, model, epoch):
            logging.info(f"Early stopping triggered at epoch {epoch}")
            # 早停时，额外绘一次图，确保捕捉到最后状态
            exp_manager.render_plot(layer_idx)
            break
    
    if args.early_stop:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    return model

def update_boosting(model, features, labels, train_mask, sample_weights, cost_matrix, n_classes, g, args, device):
    model.eval()
    with torch.no_grad():
        ada_use_h, _, attention = model(features)

    output_logp = torch.log(F.softmax(ada_use_h, dim=1))
    h = (n_classes - 1) * (output_logp - torch.mean(output_logp, dim=1).view(-1, 1))
    
    y_pred = torch.argmax(h[train_mask], dim=1)
    cost = torch.tensor(misclassification_cost(labels[train_mask].cpu(), y_pred.cpu(), cost_matrix), device=device)
    
    temp = F.nll_loss(F.log_softmax(ada_use_h[train_mask], 1), labels[train_mask], reduction='none')
    estimator_weight = (n_classes - 1) / n_classes * temp
    
    new_weights = sample_weights * torch.exp(estimator_weight * cost * ((sample_weights > 0) | (estimator_weight < 0)))
    sample_weights = (new_weights / new_weights.sum()).detach()

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

    if args.blank == 1: sys.exit()

    g, features, labels, n_classes = load_data(args, device)
    train_mask, test_mask = g.ndata['train_mask'], g.ndata['test_mask']

    final_results = torch.zeros(g.num_nodes(), n_classes).to(device)
    sample_weights = torch.ones(train_mask.sum().item(), device=device)
    sample_weights = sample_weights / sample_weights.sum()

    how_dic = {0:'uniform', 1:'inverse', 2:'log1p-inverse'}
    cost_matrix_df = cost_table_calc(_validate_cost_matrix(_set_cost_matrix(labels[train_mask].cpu(), how=how_dic[args.cost]), n_classes))
    stopper = EarlyStopping(args.patience) if args.early_stop else None

    # 模型定义在循环外部 (Boosting 微调模式)
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT_COBO(g, args.num_layers, features.shape[1], args.hid, n_classes, heads, F.elu, 
                        args.dropout, args.adj_dropout, args.in_drop, args.attn_drop, args.negative_slope, args.residual).to(device)

    for layer in range(args.layers):
        if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()
        logging.info(f"|This is the {layer + 1}th layer!|")

        # 优化器重置
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        
        # Train
        model = train_single_layer(layer, model, optimizer, g, features, labels, sample_weights, n_classes, args, stopper, exp_manager)
        
        # Boosting Update
        h, sample_weights, features = update_boosting(model, features, labels, train_mask, sample_weights, cost_matrix_df, n_classes, g, args, device)
        final_results += h

    # Final Evaluation
    test_logits = final_results[test_mask]
    if torch.isnan(test_logits).any(): test_logits = torch.nan_to_num(test_logits)
    
    t_acc, t_f1, t_rec, t_gmean, t_auc = calculate_metrics(test_logits, labels[test_mask], n_classes)

    preds = torch.argmax(test_logits, dim=1).cpu().numpy()
    labels_np = labels[test_mask].cpu().numpy()
    report = classification_report(labels_np, preds, digits=4)
    logging.info(f"\nFinal Test Report:\n{report}")
    
    result_str = (f"Final Result -> "
                  f"Macro AUC: {t_auc:.4f}, "
                  f"Macro F1: {t_f1:.4f}, "
                  f"Macro Recall: {t_rec:.4f}, "
                  f"G-mean: {t_gmean:.4f}, "
                  f"Acc: {t_acc:.4f}")
    logging.info(result_str)
    
    final_metrics = {
        "Macro AUC": t_auc,
        "Macro F1": t_f1,
        "Macro Recall": t_rec,
        "G-mean": t_gmean,
        "Accuracy": t_acc
    }
    exp_manager.save_experiment_summary(args, final_metrics)

if __name__ == "__main__":
    main()