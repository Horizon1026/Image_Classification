# Image Classification

CIFAR-10 classification example using the sibling `Perception_Utility` framework. The model is defined in [`src/models/cifar_cnn.py`](src/models/cifar_cnn.py); data, task, and training components live in `Perception_Utility/src`.

Dependencies: Python 3.10+, PyTorch, torchvision, NumPy, Pillow, TensorBoard (`pip install tensorboard`). Dataset layout: `Cifar10/{train,test}/{class_name}/*.png`.

```bash
./run.sh --epochs 10
```

The default dataset path is `/media/horizon/Database/robotic_datasets/visual_learning/Cifar10`. Override it with `--data-root`. The last checkpoint is written to `output/last.ckpt`. Training applies configurable padded random crop and horizontal flip before normalization; validation only normalizes images. Both pipelines operate on complete samples, and classification labels remain unchanged.

Short smoke run:

```bash
./run.sh --epochs 1 --max-train-samples 64 --max-val-samples 32 --batch-size 16 --num-workers 0
./run.sh --epochs 2 --max-train-samples 64 --max-val-samples 32 --batch-size 16 --num-workers 0 --resume
```

`--epochs` is the total number of epochs, including completed epochs when resuming. Set `PYTHON_BIN` to an environment's Python path if needed.

From `Workspace/scripts`, the same experiment can be started with `sh run_test.sh Image_Classification` or `./run_test.sh Image_Classification`. Extra arguments are passed through, for example `--epochs 1 --max-train-samples 64`.

Training starts a local dashboard at `http://127.0.0.1:8765/` and prints its URL in the terminal. Open the URL manually while training runs. Use `--dashboard-port` to choose another port when needed. It polls for current samples, predictions, loss, and metrics every two seconds by default:

```bash
./run.sh --refresh-seconds 2
```

Use `--refresh-seconds` to change the polling and time-based logging interval. The page is available while training runs; after training, saved TensorBoard events remain in `output/tensorboard` and can be viewed with:

```bash
tensorboard --logdir output/tensorboard
```

Use `--no-live-dashboard` to keep only TensorBoard events. `--scalar-log-interval`, `--image-log-interval`, `--preview-images`, and `--log-dir` adjust logging details. `--no-visualization` disables all visualization. Resume training with the same log directory to continue the curves.

The live charts label training steps or epochs on the horizontal axis. Cross-entropy uses nats per sample; accuracy, macro precision, macro recall, and macro F1 are displayed as percentages. The underlying metric values remain fractions in TensorBoard logs.

The live page has separate **Loss**, **Metrics**, and **Learning rate** sections. Each plot pairs Train on the left with Val on the right; an unavailable counterpart leaves an empty slot, including the Val slot for learning rate.
