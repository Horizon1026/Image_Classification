import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from augmentor import ClassificationAugment, ComposeSampleTransforms, NormalizeImage
from core import get_valid_device, seed_everything
from data import ImageFolderDataset, classification_collate
from engine import Trainer
from tasks import ClassificationTask
from visuializor import (
    DEFAULT_DASHBOARD_PORT, ClassificationPreview, build_training_visualizer,
)
from models import CifarCNN


# Use the local CIFAR-10 folder unless another root is supplied.
DEFAULT_DATA_ROOT = Path("/media/horizon/Database/robotic_datasets/visual_learning/Cifar10")

# Define chart labels and display units for this classification experiment.
METRIC_DISPLAY = {
    "loss": {"label": "Cross-entropy loss", "unit": "nats/sample", "scale": 1},
    "accuracy": {"label": "Accuracy", "unit": "%", "scale": 100},
    "precision_macro": {"label": "Macro precision", "unit": "%", "scale": 100},
    "recall_macro": {"label": "Macro recall", "unit": "%", "scale": 100},
    "f1_macro": {"label": "Macro F1", "unit": "%", "scale": 100},
    "learning_rate": {"label": "Learning rate", "unit": "unitless", "scale": 1},
}


# Parse the experiment configuration from the command line.
def parse_args():
    # Create the command-line parser for this CIFAR-10 experiment.
    parser = argparse.ArgumentParser(description="Train a CIFAR-10 classifier")
    # Locate the dataset root containing train and test class folders.
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    # Set the total epoch count, including epochs completed before resume.
    parser.add_argument("--epochs", type=int, default=100)
    # Set the number of samples in each training and validation batch.
    parser.add_argument("--batch-size", type=int, default=128)
    # Set the initial AdamW rate, which a resumed optimizer will override.
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    # Select cosine annealing or keep the learning rate constant.
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
    # Set the cosine half-cycle length in completed epochs.
    parser.add_argument("--cosine-t-max", type=int, default=100)
    # Set the learning-rate floor reached at the end of the cosine cycle.
    parser.add_argument("--min-learning-rate", type=float, default=0.0)
    # Set the number of data-loading worker processes per loader.
    parser.add_argument("--num-workers", type=int, default=4)
    # Set how many training batches share one optimizer update.
    parser.add_argument("--accumulation-steps", type=int, default=1)
    # Select CPU or CUDA explicitly, or detect an available device.
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    # Seed Python, NumPy, PyTorch, and deterministic subset selection.
    parser.add_argument("--seed", type=int, default=42)
    # Set the path used to save checkpoints and load them on resume.
    parser.add_argument("--checkpoint", type=Path, default=Path("output/last.ckpt"))
    # Restore model, optimizer, progress, and random states from a checkpoint.
    parser.add_argument("--resume", action="store_true")
    # Limit training to a fixed, seeded subset for short runs.
    parser.add_argument("--max-train-samples", type=int, default=None)
    # Limit validation to a separate fixed, seeded subset.
    parser.add_argument("--max-val-samples", type=int, default=None)
    # Log training scalars after this many batches or the refresh interval.
    parser.add_argument("--scalar-log-interval", type=int, default=20)
    # Log training previews after this many batches or the refresh interval.
    parser.add_argument("--image-log-interval", type=int, default=500)
    # Cap the number of samples shown in each preview image.
    parser.add_argument("--preview-images", type=int, default=8)
    # Set page polling and time-based logging intervals.
    parser.add_argument("--refresh-seconds", type=float, default=2.0)
    # Bind the local dashboard to this port, or use 0 for a temporary port.
    parser.add_argument("--dashboard-port", type=int, default=DEFAULT_DASHBOARD_PORT)
    # Disable the live page while keeping the binary log.
    parser.add_argument("--no-live-dashboard", action="store_true")
    # Store the binary log of the desktop viewer in this file when visualization is enabled.
    parser.add_argument("--binlog", type=Path, default=Path("output/train.binlog"))
    # Disable the binary log file while keeping the other visualization targets.
    parser.add_argument("--no-binlog", action="store_true")
    # Disable both binary logging and the live page.
    parser.add_argument("--no-visualization", action="store_true")

    # Parse the supplied command-line values into an argument namespace.
    return parser.parse_args()


# Select a repeatable subset for short training runs.
def limit_dataset(dataset, limit, seed):
    # Keep the full dataset when no limit is requested.
    if limit is None:
        return dataset
    if not 1 <= limit <= len(dataset):
        raise ValueError(f"sample limit must be between 1 and {len(dataset)}")
    # Sample indices with an isolated, deterministic generator.
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:limit].tolist()
    return Subset(dataset, indices)


# Print every parsed option and the resolved experiment configuration.
def report_configuration(args, train_dataset, val_dataset, train_loader, val_loader, trainer):
    prefix = "[ImageClassification]"
    lines = [f"{prefix} Configuration", f"{prefix} Command-line arguments:"]

    # Include every declared argument in parser order.
    for name, value in vars(args).items():
        option = "--" + name.replace("_", "-")
        lines.append(f"{prefix}   {option}: {value}")

    # Report actual data, model, and optimizer state after checkpoint recovery.
    model = trainer.model
    optimizer = trainer.optimizer
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    train_transforms = " -> ".join(type(item).__name__ for item in train_dataset.transform.transforms)
    test_transforms = " -> ".join(type(item).__name__ for item in val_dataset.transform.transforms)
    lines.extend([
        f"{prefix} Resolved setup:",
        f"{prefix}   device: {trainer.device}",
        f"{prefix}   train_split: {args.data_root / 'train'}",
        f"{prefix}   train_samples: {len(train_loader.dataset):,} selected / {len(train_dataset):,} available",
        f"{prefix}   test_split: {args.data_root / 'test'}",
        f"{prefix}   test_samples: {len(val_loader.dataset):,} selected / {len(val_dataset):,} available",
        f"{prefix}   train_transforms: {train_transforms}",
        f"{prefix}   test_transforms: {test_transforms}",
        f"{prefix}   classes: {train_dataset.classes}",
        f"{prefix}   class_to_idx: {train_dataset.class_to_idx}",
        f"{prefix}   model: {type(model).__name__}",
        f"{prefix}   parameters: {total_parameters:,} total / {trainable_parameters:,} trainable",
        f"{prefix}   optimizer: {type(optimizer).__name__}",
        f"{prefix}   scheduler: {type(trainer.scheduler).__name__ if trainer.scheduler is not None else 'None'}",
        f"{prefix}   active_learning_rate: {optimizer.param_groups[0]['lr']}",
        f"{prefix}   next_epoch: {trainer.next_epoch + 1}",
    ])
    # Report effective scheduler settings, which a checkpoint may override.
    if trainer.scheduler is not None:
        lines.extend([
            f"{prefix}   active_cosine_t_max: {trainer.scheduler.T_max}",
            f"{prefix}   active_min_learning_rate: {trainer.scheduler.eta_min}",
        ])
    print("\n".join(lines), flush=True)


# Build and run the complete CIFAR-10 experiment.
def main():
    # Seed the experiment and resolve its requested device.
    args = parse_args()
    # Reject invalid schedule settings before loading the dataset.
    if args.scheduler == "cosine" and (
        args.cosine_t_max < 1 or not 0 <= args.min_learning_rate <= args.learning_rate
    ):
        raise ValueError("cosine annealing requires a positive T_max and a learning-rate floor within [0, initial rate]")
    seed_everything(args.seed)
    device = get_valid_device(args.device)

    # Compose separate sample pipelines for training and validation.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    train_transform = ComposeSampleTransforms([ClassificationAugment(), NormalizeImage(mean, std)])
    val_transform = ComposeSampleTransforms([NormalizeImage(mean, std)])
    train_dataset = ImageFolderDataset(args.data_root / "train", transform=train_transform)
    val_dataset = ImageFolderDataset(args.data_root / "test", transform=val_transform)
    # Verify that both splits use the same ten labels.
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and test class mappings differ")
    if len(train_dataset.classes) != 10:
        raise ValueError(f"Expected 10 classes, found {len(train_dataset.classes)}")

    # Build independent training and evaluation loaders.
    train_loader = DataLoader(
        limit_dataset(train_dataset, args.max_train_samples, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=classification_collate,
    )
    val_loader = DataLoader(
        limit_dataset(val_dataset, args.max_val_samples, args.seed + 1),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=classification_collate,
    )
    # Construct the model, optimizer, and reusable trainer.
    model = CifarCNN(num_classes=len(train_dataset.classes))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Create the requested native PyTorch schedule before checkpoint recovery.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cosine_t_max, eta_min=args.min_learning_rate,
        )
        if args.scheduler == "cosine" else None
    )
    trainer = Trainer(
        model, ClassificationTask(num_classes=len(train_dataset.classes)),
        optimizer, device, args.accumulation_steps, scheduler=scheduler,
    )
    # Restore full training state when continuation is requested.
    if args.resume:
        if not args.checkpoint.is_file():
            raise FileNotFoundError(args.checkpoint)
        trainer.load_checkpoint(args.checkpoint)
        print(f"[ImageClassification] Resumed from {args.checkpoint} at epoch {trainer.next_epoch}", flush=True)
    # Attach binary logging and task-aware image previews after resume.
    if not args.no_visualization and (not args.no_binlog or not args.no_live_dashboard):
        preview = ClassificationPreview(train_dataset.classes, mean, std, args.preview_images)
        trainer.visualizer = build_training_visualizer(
            None if args.no_binlog else args.binlog,
            step_metric_names=trainer.task.step_metric_names,
            epoch_metric_names=trainer.task.epoch_metric_names,
            preview=preview,
            metric_display=METRIC_DISPLAY,
            refresh_seconds=args.refresh_seconds,
            scalar_interval=args.scalar_log_interval,
            image_interval=args.image_log_interval,
            live_dashboard=not args.no_live_dashboard,
            dashboard_port=args.dashboard_port,
        )
    # Report all requested settings and resolved runtime details before training.
    report_configuration(args, train_dataset, val_dataset, train_loader, val_loader, trainer)
    try:
        trainer.fit(train_loader, val_loader, epochs=args.epochs, checkpoint_path=args.checkpoint)
    finally:
        # Flush and close visualization files after training or failure.
        if trainer.visualizer is not None:
            trainer.visualizer.close()


# Run the experiment only when invoked as a script.
if __name__ == "__main__":
    main()
