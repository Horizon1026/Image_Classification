import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader, DistributedSampler

# Define paths. Add Perception_Utility to path.
current_file_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.abspath(os.path.join(current_file_dir, "../"))
project_dir = os.path.abspath(os.path.join(repo_dir, "../"))
sys.path.append(os.path.join(project_dir, "Perception_Utility/src"))
sys.path.append(os.path.join(repo_dir, "src"))

# Import modules from current repo.
from cnn.model import CnnNet
from mlp.model import MlpNet
from resnet.model import ResNet
from vit.model import ViTNet
# Import modules from Perception_Utility.
from devices.ddp_devices import DDPHandler
from trainer.base_trainer import BaseTrainer
from datasets.mnist_dataset import MnistDataset
from datasets.cifar10_dataset import Cifar10Dataset
from visualizors.classification_visualizor import ClassificationVisualizor
from criterions.classification_criterion import CrossEntropyLoss
from metrics.classification_metrics import F1Metric

# Main function.
def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Type of model to train on classification dataset.")
    parser.add_argument("--model", type=str, default="mlp", choices=["cnn", "mlp", "resnet", "vit"],
                        help="Choose model architecture: [cnn, mlp, resnet, vit]")
    parser.add_argument("--dataset_dir", type=str, default="/media/horizon/Database/robotic_datasets/visual_learning",
                        help="Path to dataset directory.")
    parser.add_argument("--dataset_name", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Choose dataset name: [mnist, cifar10]")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train.")
    parser.add_argument("--pretrained_model_path", type=str, default=os.path.join(repo_dir, "output/final_model.pth"),
                        help="Path to pretrained model.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size per GPU.")
    parser.add_argument("--enable_distributed", action="store_true", help="Enable distributed training.")
    parser.add_argument("--use_amp", action="store_true", help="Use Automatic Mixed Precision (AMP).")
    parser.add_argument("--amp_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2"],
                        help="Data type for AMP: [float16, bfloat16, float8_e4m3fn, float8_e5m2]")
    args = parser.parse_args()

    # Setup distributed training using DDPHandler.
    handler = DDPHandler()
    if args.enable_distributed:
        handler.init_process_group(force_distributed=True)

    device = handler.get_device()
    world_size = handler.get_world_size()

    if handler.is_main_process():
        print(f"\033[93m[INFO] Test training model {args.model} on {args.dataset_name} dataset.\033[0m")
        print(f"[INFO] Training will be performed on {device}. Enable distributed training: {handler.is_distributed}")

    # Setup Augmentor.
    train_augmentor = None
    test_augmentor = None

    # Setup Datasets and Dataloaders.
    image_shape = []
    num_of_labels = 0
    if args.dataset_name == "mnist":
        num_of_labels = 10
        image_shape = [1, 28, 28]
        train_dataset = MnistDataset(args.dataset_dir, phase="train", augmentor=train_augmentor)
        test_dataset = MnistDataset(args.dataset_dir, phase="test", augmentor=test_augmentor)
    elif args.dataset_name == "cifar10":
        num_of_labels = 10
        image_shape = [3, 32, 32]
        train_dataset = Cifar10Dataset(args.dataset_dir, phase="train", augmentor=train_augmentor)
        test_dataset = Cifar10Dataset(args.dataset_dir, phase="test", augmentor=test_augmentor)

    # Setup Distributed Sampler if needed.
    train_sampler = DistributedSampler(train_dataset) if handler.is_distributed else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if handler.is_distributed else None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                              num_workers=4, pin_memory=True, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True, sampler=test_sampler)

    # Setup Model.
    if args.model == "cnn":
        if handler.is_main_process(): print("[INFO] Model: CNN.")
        model = CnnNet(image_size=image_shape, num_classes=num_of_labels)
    elif args.model == "mlp":
        if handler.is_main_process(): print("[INFO] Model: MLP.")
        model = MlpNet(image_size=image_shape, dim_hidden_layer=128, num_classes=num_of_labels)
    elif args.model == "resnet":
        if handler.is_main_process(): print("[INFO] Model: ResNet.")
        model = ResNet(in_channels=image_shape[0], num_classes=num_of_labels)
    elif args.model == "vit":
        if handler.is_main_process(): print("[INFO] Model: ViT.")
        patch_size = [8, 8]
        model = ViTNet(
            image_size = image_shape,
            patch_size = patch_size,
            dim_token = image_shape[0] * patch_size[0] * patch_size[1],
            dim_hidden_layer = 128,
            num_heads = 1,
            num_layers = 3,
            num_classes = num_of_labels,
            dropout = 0,
            use_class_token = True,
        )

    if handler.is_main_process():
        num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[INFO] Number of model parameters: {num_of_params / 1e6} M.")

    # Setup metric.
    metric = F1Metric()
    # Setup Loss.
    criterion = CrossEntropyLoss()
    # Setup Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01 * world_size, momentum=0.9, weight_decay=1e-4) # Scaling lr by world_size.
    # Setup Visualizor.
    visualizor = ClassificationVisualizor() if handler.is_main_process() else None
    # Setup Trainer.
    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        metric=metric,
        visualizor=visualizor,
        use_amp=args.use_amp,
        amp_dtype=args.amp_dtype,
        output_dir=os.path.join(repo_dir, "output/"),
    )
    # Start Training.
    if handler.is_main_process():
        print(f"[INFO] Starting training loop using Perception Utility framework.")
    trainer.train(args.max_epochs, train_loader, test_loader, pretrained_model_path=args.pretrained_model_path)
    # Cleanup.
    handler.cleanup()

if __name__ == "__main__":
    main()
