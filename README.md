# Image Classification

基于 `Perception_Utility` 的图像分类训练示例。项目分别提供 CIFAR-10 的 RGB CNN 和 MNIST PNG 的灰度图 CNN；数据集、训练接口和指标定义复用同级框架。

## Components

- [x] `src/models/cifar_cnn.py`：CIFAR-10 的三通道分类模型。
- [x] `src/models/mnist_cnn.py`：MNIST 的单通道分类模型。
- [x] `src/train.py`：按数据集选择模型和 spec，完成增强、训练/验证、配置报告、checkpoint 恢复及余弦退火。
- [x] 两个数据集均为 10 个参与训练的类别，没有“其他”类或忽略类；训练启动时打印原始类别到模型输出 ID 的映射表。
- [x] 可视化：交叉熵、准确率、宏平均 precision/recall/F1，以及图像与真值/预测预览。CIFAR-10 和 MNIST 默认记录全部 10 类的 precision/recall/F1，实时页面将这些单类曲线默认折叠，点击后展开。

## Dependencies

### Project repositories

- `Perception_Utility`：分类数据集 spec、增强、loss、指标、任务、训练器与可视化组件。
- `Binary_Data_Log/py_src`：保存二进制日志时使用；`Binary_Data_Viewer` 可用于查看日志。

以上项目需与本项目位于同级目录；`run.sh` 会将所需源码路径加入 `PYTHONPATH`。

### Python packages

- Python 3.10+、PyTorch、torchvision、NumPy、Pillow。

### Datasets

- CIFAR-10 默认根目录：`/media/horizon/Database/robotic_datasets/visual_learning/Cifar10`，目录结构为 `{train,test}/{class_name}/图片`。
- MNIST PNG 默认根目录：`/media/horizon/Database/robotic_datasets/visual_learning/MNIST/png`，目录结构为 `{training,testing}/{0..9}/图片`。

使用 `--data-root` 可覆盖所选数据集的根目录。数据集类别、通道模式和归一化统计定义在 `Perception_Utility/src/data/` 中。

## Run

在本项目根目录执行；默认数据集为 CIFAR-10：

```bash
./run.sh --dataset cifar10 --epochs 20
./run.sh --dataset mnist --epochs 20
```

如果系统默认 `python3` 没有安装所需包，可指定解释器：

```bash
PYTHON_BIN=/path/to/python ./run.sh --dataset mnist
```

快速检查训练流程：

```bash
./run.sh --dataset mnist --epochs 1 --batch-size 16 --num-workers 0 --max-train-samples 64 --max-val-samples 32 --no-visualization
```

默认 batch size 为 128，学习率为 `1e-3`。CIFAR-10 训练使用随机裁剪与水平翻转；MNIST 只做归一化，验证集也只做归一化。

## Tips

- 默认启用余弦退火：`--cosine-t-max 100`、`--min-learning-rate 0`；`--scheduler none` 保持学习率不变。恢复训练时用 `--resume`，且 `--epochs` 表示包含已完成 epoch 的总数。
- CIFAR-10 默认 checkpoint 和日志为 `output/last.ckpt`、`output/train.binlog`；MNIST 使用 `output/mnist/last.ckpt`、`output/mnist/train.binlog`。可用 `--checkpoint`、`--binlog` 改路径。
- 实时页面默认运行在 `http://127.0.0.1:8765/`。使用 `--no-live-dashboard` 仅保存日志，`--no-binlog` 仅查看实时页面，`--no-visualization` 关闭两者。
- `--max-train-samples` 与 `--max-val-samples` 使用固定随机种子选取子集；`--accumulation-steps` 可设置梯度累积步数。完整参数列表可运行 `./run.sh --help`。
