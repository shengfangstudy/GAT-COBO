environment.yaml是conda环境，conda env create -f environment.yaml，即可在另一台机器上复现出完全一致的环境。

有些包是通过 pip install 装的（比如 torchdata、某些特定版本的 dgl），也单独保存一份 pip 清单在requirements.txt

另外：
python - <<'PY'
import torch, dgl, platform, sys, os
print("Python:", sys.version)
print("Platform:", platform.platform())
print("Torch:", torch.__version__)
print("DGL:", dgl.__version__)
print("CUDA version:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU device name:", torch.cuda.get_device_name(0))
print("Conda prefix:", os.environ.get("CONDA_PREFIX", "N/A"))
PY
的输出为：
Python: 3.11.10 (main, Oct  3 2024, 07:29:13) [GCC 11.2.0]
Platform: Linux-5.4.0-150-generic-x86_64-with-glibc2.27
Torch: 2.1.1+cu121
DGL: 2.0.0+cu121
CUDA version: 12.1
CUDA available: True
GPU device name: Tesla V100-PCIE-32GB
Conda prefix: /home/lcq/ENTER/envs/lcq
