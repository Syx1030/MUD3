# CCAC Baseline

这是用于 CCAC 评测的 RNN baseline 代码。

## 环境要求

推荐使用 Conda 创建独立环境：

```bash
conda create -n ccac-baseline python=3.10 -y
conda activate ccac-baseline
python -m pip install --upgrade pip
pip install -r requirement.txt
```

如果安装时提示找不到 `torch==2.5.1+cu121`，说明 pip 只查了普通 PyPI 镜像，没有查 PyTorch CUDA wheel 源。可以强制先单独安装 PyTorch：

```bash
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26.4 pandas==2.2.3 scikit-learn==1.5.2 PyYAML==6.0.2 tqdm==4.66.5
```

`requirement.txt` 中的 PyTorch 版本为：

```text
torch==2.5.1+cu121
```

安装后可以检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

如果输出类似下面结果，说明 PyTorch 安装正常：

```text
2.5.1+cu121
12.1
True
```

如果最后一行是 `False`，说明当前环境没有检测到可用 GPU，可以先检查 `nvidia-smi`、集群 GPU 申请参数或容器/作业环境。

## 数据格式

训练入口默认读取 `--data_dir` 指向的数据目录。目录中需要包含：

```text
dep_feat.pkl
nondep_feat.pkl
labels.csv
```

其中：

- `dep_feat.pkl`：抑郁样本特征。
- `nondep_feat.pkl`：非抑郁样本特征。
- `labels.csv`：至少包含 `names` 和 `split` 两列，`split` 使用 `train`、`val`、`test`。

代码会从每个用户样本中最多读取前 360 个视频片段，每个片段按 60 帧、161 维特征进行裁剪或补齐。
```

## 运行训练

在仓库根目录运行：

```bash
cd /home/yfhou/CCAC-baseline
conda activate ccac-baseline

python main_early_ori.py \
  --data_dir /home/yfhou/CCAC-baseline/MUD3_final \
  --model LSTM_han \
  --epochs 20 \
  --batch_size 2 \
  --learning_rate 1e-5 \
  --lr_scheduler None \
  --device cuda \
  --lant_save_name LSTM_han \
  --earliness_alpha 30 \
  --seed 110
```

如果没有 GPU，可以把 `--device cuda` 改成：

```bash
--device cpu
```

## 使用 Slurm 脚本
提供了 `run_rnn.sh` 示例。使用前建议先编辑脚本中的环境激活部分，例如：

```bash
source /home/yfhou/miniconda3/etc/profile.d/conda.sh
conda activate ccac-baseline
```

然后提交：

```bash
sbatch run_rnn.sh
```

如果集群 GPU 分区或 GPU 型号不同，需要同步修改脚本中的 `#SBATCH -p` 和 `#SBATCH --gres`。

## 输出结果

训练过程中会保存验证集 loss 最低的模型：

```text
save_dir/<model>/best_model_<model>.pt
```

测试集指标会保存到：

```text
save_dir/<model>/result.json
```

早检延迟结果会追加写入：

```text
latency_<lant_save_name>.csv
```

例如 `--lant_save_name LSTM_han` 时，输出文件为：

```text
latency_LSTM_han.csv
```

## 配置文件

`config.yaml` 中保存了一些默认参数，但推荐运行时显式传入关键参数，尤其是：

- `--data_dir`
- `--model`
- `--device`
- `--epochs`
- `--batch_size`
- `--learning_rate`

这样可以避免不同机器或集群作业环境中的默认配置不一致。
