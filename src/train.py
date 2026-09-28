import argparse
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from augmentor import ImageClassificationAugment, ComposeSampleTransforms, NormalizeImage
from core import resolve_torch_device, seed_global_rngs
from data import CIFAR10_SPEC, MNIST_SPEC, image_classification_collate
from engine import Trainer
from model import format_model_parameters
from objectives import ClassificationCrossEntropyObjective
from tasks import ImageClassificationTask
from visualization import (
    CLASSIFICATION_TRAINING_SCALAR_DISPLAY, DEFAULT_DASHBOARD_PORT, ImageClassificationPreview, build_training_visualizer,
)
from models import CifarCNN, MnistCNN


# Model architecture is an experiment choice, separate from dataset metadata.
MODEL_BY_DATASET = {"cifar10": CifarCNN, "mnist": MnistCNN}
DATASET_SPECS = {"cifar10": CIFAR10_SPEC, "mnist": MNIST_SPEC}


# Parse the experiment configuration from the command line.
def parse_args():
    # Expose only datasets supported by a local model and shared data definition.
    parser = argparse.ArgumentParser(description="Train an image classifier")
    parser.add_argument("--dataset", choices=tuple(MODEL_BY_DATASET), default="cifar10")
    # Override the selected dataset's default root.
    parser.add_argument("--data-root", type=Path, default=None)
    # Set the total epoch count, including epochs completed before resume.
    parser.add_argument("--epochs", type=int, default=20)
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
    parser.add_argument("--checkpoint", type=Path, default=None)
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
    parser.add_argument("--binlog", type=Path, default=None)
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
def report_configuration(args, dataset_spec, train_dataset, val_dataset, train_loader, val_loader, trainer):
    prefix = "[ImageClassification]"
    lines = [f"{prefix} Configuration", f"{prefix} Command-line arguments:"]

    # Include every declared argument in parser order.
    for name, value in vars(args).items():
        option = "--" + name.replace("_", "-")
        lines.append(f"{prefix}   {option}: {value}")

    # Report actual data, model, and optimizer state after checkpoint recovery.
    model = trainer.model
    optimizer = trainer.optimizer
    train_transforms = " -> ".join(type(item).__name__ for item in train_dataset.transform.transforms)
    test_transforms = " -> ".join(type(item).__name__ for item in val_dataset.transform.transforms)
    lines.extend([
        f"{prefix} Resolved setup:",
        f"{prefix}   device: {trainer.device}",
        f"{prefix}   train_split: {dataset_spec.split_path('train')}",
        f"{prefix}   train_samples: {len(train_loader.dataset):,} selected / {len(train_dataset):,} available",
        f"{prefix}   test_split: {dataset_spec.split_path('val')}",
        f"{prefix}   test_samples: {len(val_loader.dataset):,} selected / {len(val_dataset):,} available",
        f"{prefix}   train_transforms: {train_transforms}",
        f"{prefix}   test_transforms: {test_transforms}",
        f"{prefix}   model: {type(model).__name__}",
        f"{prefix}   parameters: {format_model_parameters(model)}",
        f"{prefix}   optimizer: {type(optimizer).__name__}",
        f"{prefix}   scheduler: {type(trainer.scheduler).__name__ if trainer.scheduler is not None else 'None'}",
        f"{prefix}   active_learning_rate: {optimizer.param_groups[0]['lr']}",
        f"{prefix}   next_epoch: {trainer.next_epoch + 1}",
    ])
    # Show each dataset source class beneath its model output index.
    lines.append(f"{prefix} Label mapping:")
    lines.extend(f"{prefix}   {row}" for row in dataset_spec.label_table_lines(train_dataset.source_class_to_idx))
    # Report effective scheduler settings, which a checkpoint may override.
    if trainer.scheduler is not None:
        lines.extend([
            f"{prefix}   active_cosine_t_max: {trainer.scheduler.T_max}",
            f"{prefix}   active_min_learning_rate: {trainer.scheduler.eta_min}",
        ])
    print("\n".join(lines), flush=True)


# Build and run the selected image classification experiment.
def main():
    args = parse_args()
    # Resolve the shared dataset definition and any machine-specific root override.
    dataset_spec = DATASET_SPECS[args.dataset]
    if args.data_root is not None:
        dataset_spec = replace(dataset_spec, root=args.data_root)
    args.data_root = dataset_spec.root
    # Keep default checkpoints and binary logs separate across datasets.
    if args.checkpoint is None:
        args.checkpoint = Path("output/last.ckpt" if args.dataset == "cifar10" else f"output/{args.dataset}/last.ckpt")
    if args.binlog is None:
        args.binlog = Path("output/train.binlog" if args.dataset == "cifar10" else f"output/{args.dataset}/train.binlog")
    # Reject invalid schedule settings before loading the dataset.
    if args.scheduler == "cosine" and (
        args.cosine_t_max < 1 or not 0 <= args.min_learning_rate <= args.learning_rate
    ):
        raise ValueError("cosine annealing requires a positive T_max and a learning-rate floor within [0, initial rate]")
    seed_global_rngs(args.seed)
    device = resolve_torch_device(args.device)

    # Apply digit-safe or CIFAR-specific augmentation before normalization.
    mean, std = dataset_spec.mean, dataset_spec.std
    train_operations = [ImageClassificationAugment()] if args.dataset == "cifar10" else []
    train_transform = ComposeSampleTransforms([*train_operations, NormalizeImage(mean, std)])
    val_transform = ComposeSampleTransforms([NormalizeImage(mean, std)])
    # Build both splits through the shared dataset contract.
    train_dataset = dataset_spec.make_dataset("train", transform=train_transform)
    val_dataset = dataset_spec.make_dataset("val", transform=val_transform)
    # Verify that both splits use the same source folders and target classes.
    if train_dataset.source_class_to_idx != val_dataset.source_class_to_idx:
        raise ValueError("Training and test source class mappings differ")
    if train_dataset.classes != dataset_spec.classes.names or val_dataset.classes != dataset_spec.classes.names:
        raise ValueError("Dataset target classes differ from the spec")

    # Build independent training and evaluation loaders.
    train_loader = DataLoader(
        limit_dataset(train_dataset, args.max_train_samples, args.seed),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=image_classification_collate,
    )
    val_loader = DataLoader(
        limit_dataset(val_dataset, args.max_val_samples, args.seed + 1),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=image_classification_collate,
    )
    # Construct the model, optimizer, and reusable trainer.
    model = MODEL_BY_DATASET[args.dataset](num_classes=dataset_spec.classes.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    # Create the requested native PyTorch schedule before checkpoint recovery.
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.cosine_t_max, eta_min=args.min_learning_rate,
        )
        if args.scheduler == "cosine" else None
    )
    trainer = Trainer(
        model, ImageClassificationTask(num_classes=dataset_spec.classes.num_classes, objective=ClassificationCrossEntropyObjective()),
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
        preview = ImageClassificationPreview(dataset_spec.classes.names, mean, std, args.preview_images)
        trainer.visualizer = build_training_visualizer(
            None if args.no_binlog else args.binlog,
            step_metric_names=trainer.task.step_metric_names,
            epoch_metric_names=trainer.task.epoch_metric_names,
            preview=preview,
            metric_display=CLASSIFICATION_TRAINING_SCALAR_DISPLAY,
            refresh_seconds=args.refresh_seconds,
            scalar_interval=args.scalar_log_interval,
            image_interval=args.image_log_interval,
            live_dashboard=not args.no_live_dashboard,
            dashboard_port=args.dashboard_port,
        )
    # Report all requested settings and resolved runtime details before training.
    report_configuration(args, dataset_spec, train_dataset, val_dataset, train_loader, val_loader, trainer)
    try:
        trainer.fit(train_loader, val_loader, epochs=args.epochs, checkpoint_path=args.checkpoint)
    finally:
        # Flush and close visualization files after training or failure.
        if trainer.visualizer is not None:
            trainer.visualizer.close()


# Run the experiment only when invoked as a script.
if __name__ == "__main__":
    main()
