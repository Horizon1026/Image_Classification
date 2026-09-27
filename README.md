# Image Classification

CIFAR-10 and MNIST PNG classification examples using the sibling `Perception_Utility` framework. Their models live in `src/models/`; shared dataset definitions, data, objective, task, and training components live in `Perception_Utility/src`. The training entry point selects a shared dataset definition and pairs it with a local model.

Dependencies: Python 3.10+, PyTorch, torchvision, NumPy, Pillow, and the sibling `Binary_Data_Log/py_src` Python recorder for binary logging. `run.sh` adds that recorder to `PYTHONPATH`; no third-party logging package is needed. Dataset layouts: `Cifar10/{train,test}/{class_name}/*.png` and `MNIST/png/{training,testing}/{0..9}/*.png`.

```bash
./run.sh --epochs 10
```

`--dataset cifar10` is the default and uses `/media/horizon/Database/robotic_datasets/visual_learning/Cifar10`. Select the grayscale digit dataset with `--dataset mnist`; its default root is `/media/horizon/Database/robotic_datasets/visual_learning/MNIST/png`. Override either root with `--data-root`. CIFAR keeps its checkpoint and binary log defaults under `output/`; MNIST uses `output/mnist/` so runs do not share model checkpoints. CIFAR training uses padded random crop and horizontal flip; MNIST only normalizes images because horizontal flips alter digit labels. Validation only normalizes images. Both pipelines operate on complete samples, and classification labels remain unchanged.

MNIST smoke run:

```bash
./run.sh --dataset mnist --epochs 1 --max-train-samples 64 --max-val-samples 32 --batch-size 16 --num-workers 0
```

Short smoke run:

```bash
./run.sh --epochs 1 --max-train-samples 64 --max-val-samples 32 --batch-size 16 --num-workers 0
./run.sh --epochs 2 --max-train-samples 64 --max-val-samples 32 --batch-size 16 --num-workers 0 --resume
```

`--epochs` is the total number of epochs, including completed epochs when resuming. The default PyTorch `CosineAnnealingLR` schedule lowers the learning rate from its initial value to zero over 100 epochs. Use `--cosine-t-max` to set the cycle length and `--min-learning-rate` to set its floor, or `--scheduler none` to keep the rate constant. For a short run, use `--cosine-t-max 4`; keep this value the same when resuming. Scheduler progress and configuration are stored in the checkpoint and restored by `--resume`. A checkpoint written with StepLR cannot resume under cosine annealing; start a new run for the new schedule. Set `PYTHON_BIN` to an environment's Python path if needed.

From `Workspace/scripts`, the same experiment can be started with `sh run_test.sh Image_Classification` or `./run_test.sh Image_Classification`. Extra arguments are passed through, for example `--epochs 1 --max-train-samples 64`.

Training starts a local dashboard at `http://127.0.0.1:8765/` and prints its URL in the terminal. Open the URL manually while training runs. Use `--dashboard-port` to choose another port when needed. It polls for current samples, predictions, loss, and metrics every two seconds by default:

```bash
./run.sh --refresh-seconds 2
```

Use `--refresh-seconds` to change the page polling and time-based logging interval. The page is available while training runs. Curves and previews are saved to `output/train.binlog` by default; open this file with the sibling `Binary_Data_Viewer` application after training.

Use `--binlog` to choose another file and `--no-binlog` for a live-page-only run. Use `--no-live-dashboard` to write only the binary log. `--scalar-log-interval`, `--image-log-interval`, and `--preview-images` adjust event frequency and preview size. `--no-visualization` disables both outputs. Every run creates a new binary log, so a resumed run records only its new epochs. The binary log groups scalar items under `train` and `val` packages and stores previews in separate PNG packages. Its metric timestamps use global training steps; epoch summaries and validation previews land just after the last batch of their epoch.

The live charts use the shared classification metric display preset from `Perception_Utility` and label training steps or epochs on the horizontal axis. Cross-entropy uses nats per sample; accuracy, macro precision, macro recall, and macro F1 are displayed as percentages. The underlying metric values remain fractions in the binary log.

The live page has separate **Loss**, **Metrics**, and **Learning rate** sections. Each plot pairs Train on the left with Val on the right; an unavailable counterpart leaves an empty slot, including the Val slot for learning rate.
