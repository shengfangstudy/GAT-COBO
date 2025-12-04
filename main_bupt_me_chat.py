import matplotlib
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

from sklearn.metrics import (
    f1_score,
    classification_report,
    roc_auc_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from dgl.data.utils import load_graphs, load_info
from imblearn.metrics import geometric_mean_score

# 假设 calc_dynamic_factors 已经在 utils.py 中
from utilsme1 import (
    EarlyStopping,
    calc_dynamic_factors
)
from model import GAT_COBO


# ====================== 0. 实验记录工具 ======================
class ExperimentManager:
    def __init__(self, base_dir="training_records_bupt_has"):
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
        except Exception:
            pass

    def setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.handlers = []
        formatter = logging.Formatter(
            '%(asctime)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

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
            (0, 0): ('Loss', 'train_loss', 'val_loss', 'r'),
            (0, 1): ('AUC', 'train_auc', 'val_auc', 'b'),
            (1, 0): ('F1', 'train_f1', 'val_f1', 'g'),
            (1, 1): ('Recall', 'train_recall', 'val_recall', 'm')
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
            if isinstance(o, np.generic):
                return o.item()
            raise TypeError

        with open(os.path.join(self.run_dir, "summary.json"),
                  'w', encoding='utf-8') as f:
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
    parser = argparse.ArgumentParser(description='HAS-GNN Boosting (BUPT)')
    parser.add_argument('--dataset', type=str, default='BUPT')  # BUPT or Sichuan
    parser.add_argument('--train_size', type=float, default=0.6)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--hid', type=int, default=128)

    # Dropout
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--adj_dropout", type=float, default=0.4)
    parser.add_argument("--attn_drop", type=float, default=0.5)
    parser.add_argument("--in_drop", type=float, default=0.1)

    # Boosting Coefficients
    parser.add_argument('--attention_weight', type=float, default=0.1)
    parser.add_argument('--feature_weight', type=float, default=0.1)

    # Structure
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of Boosting Layers')
    parser.add_argument("--num_layers", type=int, default=1,
                        help="GAT layers inside each booster")
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument("--num_out_heads", type=int, default=1)

    # Training
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument('--patience', type=int, default=60)
    parser.add_argument('--early_stop', action='store_true', default=False)
    parser.add_argument("--residual", action="store_true", default=False)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--att_loss_weight', type=float, default=0.5)
    parser.add_argument('--print_interval', type=int, default=50)

    # Imbalance & Cost
    parser.add_argument('--IR', type=float, default=0.1)
    parser.add_argument('--IR_set', type=int, default=0)
    parser.add_argument('--cost', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--blank', type=int, default=0)

    # HAS-GNN 超参
    parser.add_argument('--lambda_I', type=float, default=1.0,
                        help='Coef for Structural Importance')
    parser.add_argument('--lambda_G', type=float, default=1.0,
                        help='Coef for Group Difficulty')
    parser.add_argument('--lambda_D', type=float, default=1.0,
                        help='Coef for Camouflage Difficulty')

    return parser.parse_args()


# ====================== 2. 指标 & Mask ======================
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
            auc = roc_auc_score(labels_np, probs_np,
                                average='macro', multi_class='ovo')
    except Exception:
        auc = 0.0

    return acc, f1, recall, gmean, auc


def gen_mask(g, train_rate, val_rate, IR, IR_set):
    """
    兼容四川 & BUPT 的划分策略：
    - 欺诈类：label == 1
    - benign：label != 1（普通用户 + 快递员）
    """
    labels = g.ndata['label'].long().numpy()
    n_nodes = len(labels)
    index = list(range(n_nodes))

    if IR_set != 0:
        fraud_idx = np.where(labels == 1)[0].tolist()
        benign_idx = np.where(labels != 1)[0].tolist()
        current_IR = len(fraud_idx) / len(benign_idx) if len(benign_idx) > 0 else 0.0

        if IR < current_IR and len(fraud_idx) > 0:
            n_sample = int(IR * len(benign_idx))
            n_sample = max(1, min(n_sample, len(fraud_idx)))
            index = benign_idx + random.sample(fraud_idx, n_sample)
        elif len(benign_idx) > 0:
            n_sample = int(len(fraud_idx) / IR) if IR > 0 else len(benign_idx)
            n_sample = max(1, min(n_sample, len(benign_idx)))
            index = random.sample(benign_idx, n_sample) + fraud_idx

    sub_labels = labels[index]
    train_idx, val_test_idx, _, y_vt = train_test_split(
        index,
        sub_labels,
        stratify=sub_labels,
        train_size=train_rate,
        random_state=2,
        shuffle=True
    )
    val_idx, test_idx, _, _ = train_test_split(
        val_test_idx, y_vt,
        train_size=val_rate/(1-train_rate),
        random_state=2,
        shuffle=True
    )

    g.ndata['train_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['val_mask'] = torch.zeros(n_nodes, dtype=torch.bool)
    g.ndata['test_mask'] = torch.zeros(n_nodes, dtype=torch.bool)

    g.ndata['train_mask'][train_idx] = True
    g.ndata['val_mask'][val_idx] = True
    g.ndata['test_mask'][test_idx] = True
    return g


def load_data(args, device):
    if args.dataset == 'BUPT':
        dataset, _ = load_graphs("./data/BUPT_tele_me_chat.bin")
        n_classes = load_info("./data/BUPT_tele.pkl")['num_classes']
    elif args.dataset == 'Sichuan':
        dataset, _ = load_graphs("./data/Sichuan_tele_zhu.bin")
        n_classes = load_info("./data/Sichuan_tele.pkl")['num_classes']
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    g = dataset[0]
    g = gen_mask(g, args.train_size, 0.2, args.IR, args.IR_set)
    g = g.int().to(device)

    for e in g.etypes:
        g = dgl.remove_self_loop(g, etype=e)
        g = dgl.add_self_loop(g, etype=e)

    return g, g.ndata['feat'].float(), g.ndata['label'].long(), n_classes


# ====================== 3. 训练单层 ======================
def train_single_layer(layer_idx, model, optimizer, g, features, labels,
                       sample_weights, n_classes, args, stopper, exp_manager):
    train_mask = g.ndata['train_mask']
    val_mask = g.ndata['val_mask']

    if args.early_stop and stopper is not None:
        stopper.best_score = None
        stopper.counter = 0
        stopper.early_stop = False

    for epoch in range(args.epochs):
        model.train()
        logits, logits_GAT, _ = model(features)

        logits = logits.view(logits.shape[0], -1)
        logits_GAT = logits_GAT.view(logits_GAT.shape[0], -1)

        ce_node = F.cross_entropy(
            logits[train_mask], labels[train_mask], reduction='none')
        loss_node = (ce_node * sample_weights).sum()

        ce_att = F.cross_entropy(
            logits_GAT[train_mask], labels[train_mask], reduction='none')
        loss_att = args.att_loss_weight * (ce_att * sample_weights).sum()

        loss = loss_node + loss_att

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            t_acc, t_f1, t_rec, t_gmean, t_auc = calculate_metrics(
                logits[train_mask], labels[train_mask], n_classes)

            model.eval()
            v_logits, _, _ = model(features)
            v_logits = v_logits.view(v_logits.shape[0], -1)
            v_loss = F.cross_entropy(
                v_logits[val_mask], labels[val_mask]).item()
            v_acc, v_f1, v_rec, v_gmean, v_auc = calculate_metrics(
                v_logits[val_mask], labels[val_mask], n_classes)
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
            logging.info(
                f"[Layer {layer_idx+1}] Ep {epoch}: "
                f"Loss={loss.item():.4f}/{v_loss:.4f} | "
                f"AUC={t_auc:.4f}/{v_auc:.4f}"
            )

        if args.early_stop and stopper is not None:
            if stopper.step(v_loss, model, epoch):
                logging.info(f"Early stopping triggered at epoch {epoch}")
                exp_manager.render_plot(layer_idx)
                break

    if args.early_stop and stopper is not None:
        model.load_state_dict(torch.load('es_checkpoint.pt'))

    return model


# ====================== 4. HAS-GNN Boosting 更新 ======================
def update_boosting(model, features, labels, train_mask, sample_weights,
                    n_classes, g, args, device):
    """
    最终版：严格按照
    W = alpha_class * (1 + λ_I I_v) * (1 + λ_G G_v) * (1 + λ_D D_v)
    """
    model.eval()
    with torch.no_grad():
        ada_use_h, _, attention = model(features)

    # 1) 用于最终集成的 h
    output_logp = torch.log(F.softmax(ada_use_h, dim=1))
    h = (n_classes - 1) * (
        output_logp - torch.mean(output_logp, dim=1).view(-1, 1)
    )

    # 2) 重新根据 HAS-GNN 公式计算本层后的 sample_weights
    #   A. 类别静态权重
    w_class = g.ndata['class_weight'][train_mask].to(device)
    #   B. 结构静态因子 I
    I_val = g.ndata['struct_score'][train_mask].to(device)
    #   C. 动态因子 G, D
    full_G, full_D = calc_dynamic_factors(ada_use_h, g, args, device)
    G_val = full_G[train_mask]
    D_val = full_D[train_mask]

    term_struct = 1.0 + args.lambda_I * I_val
    term_group = 1.0 + args.lambda_G * G_val
    term_camo = 1.0 + args.lambda_D * D_val

    new_weights = w_class * term_struct * term_group * term_camo
    sample_weights = (new_weights / new_weights.sum()).detach()

    # 3) Attention 引导的特征更新
    attention = attention.view(attention.shape[0], -1)
    row, col = g.edges()[0].cpu().numpy(), g.edges()[1].cpu().numpy()
    att_data = attention.cpu().numpy().T.squeeze()

    adj_att = torch.sparse_coo_tensor(
        torch.tensor([row, col]).to(device),
        torch.tensor(att_data).to(device),
        (g.num_nodes(), g.num_nodes())
    )

    new_features = torch.sparse.mm(
        args.attention_weight * adj_att,
        args.feature_weight * features
    ).detach()

    return h, sample_weights, new_features


# ====================== 5. 主流程 ======================
def main():
    args = get_args()
    setup_seed(args.seed)

    exp_manager = ExperimentManager()
    exp_manager.setup_logging()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Arguments: {args}")

    if args.blank == 1:
        sys.exit()

    g, features, labels, n_classes = load_data(args, device)
    train_mask = g.ndata['train_mask']
    test_mask = g.ndata['test_mask']

    final_results = torch.zeros(g.num_nodes(), n_classes).to(device)

    # -------- 初始化权重：类别 + 结构 --------
    logging.info("Initializing weights using Static HAS-GNN Priors...")

    w_class = g.ndata['class_weight'][train_mask].to(device)
    w_struct = 1.0 + args.lambda_I * g.ndata['struct_score'][train_mask].to(device)

    init_weights = w_class * w_struct
    sample_weights = init_weights / init_weights.sum()

    stopper = EarlyStopping(args.patience) if args.early_stop else None

    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT_COBO(
        g,
        args.num_layers,
        features.shape[1],
        args.hid,
        n_classes,
        heads,
        F.elu,
        args.dropout,
        args.adj_dropout,
        args.in_drop,
        args.attn_drop,
        args.negative_slope,
        args.residual
    ).to(device)

    for layer in range(args.layers):
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        logging.info(f"|This is the {layer + 1}th layer!|")

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

        # 训练当层弱分类器
        model = train_single_layer(
            layer, model, optimizer, g, features, labels,
            sample_weights, n_classes, args, stopper, exp_manager
        )

        # Boosting 更新
        h, sample_weights, features = update_boosting(
            model, features, labels, train_mask,
            sample_weights,
            n_classes, g, args, device
        )
        final_results += h

    # -------- 最终评估 --------
    test_logits = final_results[test_mask]
    if torch.isnan(test_logits).any():
        test_logits = torch.nan_to_num(test_logits)

    t_acc, t_f1, t_rec, t_gmean, t_auc = calculate_metrics(
        test_logits, labels[test_mask], n_classes)

    preds = torch.argmax(test_logits, dim=1).cpu().numpy()
    labels_np = labels[test_mask].cpu().numpy()
    report = classification_report(labels_np, preds, digits=4)
    logging.info(f"\nFinal Test Report:\n{report}")

    result_str = (
        f"Final Result -> Macro AUC: {t_auc:.4f}, "
        f"Macro F1: {t_f1:.4f}, Macro Recall: {t_rec:.4f}, "
        f"G-mean: {t_gmean:.4f}, Acc: {t_acc:.4f}"
    )
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
