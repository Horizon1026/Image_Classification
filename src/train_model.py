import os
import sys
import torch
import argparse
from torch.utils.data import DataLoader

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
from trainer.base_trainer import BaseTrainer
from datasets.mnist_dataset import MnistDataset
from visualizors.classification_visualizor import ClassificationVisualizor

# Define label loss for classification.
class LabelLoss():
    def __call__(self, predictions, targets):
        """
        targets shape: (B, 1)
        predictions shape: (B, 10) (Linear layer direct output)
        """
        # CrossEntropyLoss expects (B, C) for predictions and (B) long for targets.
        targets = targets.long().squeeze(1)
        return torch.nn.CrossEntropyLoss()(predictions, targets)

# Main function.
def main():
    # Parse arguments.
    parser = argparse.ArgumentParser(description="Type of model to train on classification dataset.")
    parser.add_argument("--model", type=str, default="mlp", choices=["cnn", "mlp", "resnet", "vit"],
                        help="Choose model architecture: [cnn, mlp, resnet, vit]")
    parser.add_argument("--dataset_dir", type=str, default="/media/horizon/Database/robotic_datasets/visual_learning",
                        help="Path to dataset directory.")
    parser.add_argument("--dataset_name", type=str, default="mnist", choices=["mnist"],
                        help="Choose dataset name: [mnist]")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs to train.")
    parser.add_argument("--pretrained_model_path", type=str, default=os.path.join(repo_dir, "output/final_model.pth"),
                        help="Path to pretrained model.")
    args = parser.parse_args()
    print(f"\033[93m[INFO] Test training model {args.model} on {args.dataset_name} dataset.\033[0m")

    # Setup device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Training will be performed on {device}.")
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
        train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    # Setup Model.
    if args.model == "cnn":
        print("[INFO] Model: CNN.")
        model = CnnNet(
            in_channels=image_shape[0],
            num_classes=num_of_labels
        )
    elif args.model == "mlp":
        print("[INFO] Model: MLP.")
        model = MlpNet(
            image_size = image_shape,
            dim_hidden_layer = 128,
            num_classes = num_of_labels
        )
    elif args.model == "resnet":
        print("[INFO] Model: ResNet.")
        model = ResNet(
            in_channels=image_shape[0],
            num_classes=num_of_labels
        )
    elif args.model == "vit":
        print("[INFO] Model: ViT.")
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
    num_of_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Number of model parameters: {num_of_params} Bytes.")

    # Setup Loss.
    criterion = LabelLoss()
    # Setup Optimizer.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    # Setup Visualizor.
    visualizor = ClassificationVisualizor()
    # Setup Trainer.
    trainer = BaseTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        visualizor=visualizor,
        output_dir=os.path.join(repo_dir, "output/"),
    )
    # Start Training.
    print(f"[INFO] Starting training loop using Perception Utility framework.")
    trainer.train(args.max_epochs, train_loader, test_loader, pretrained_model_path=args.pretrained_model_path)

if __name__ == "__main__":
    main()
