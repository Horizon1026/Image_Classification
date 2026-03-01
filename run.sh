python3 ./src/train_model.py --model mlp --dataset_dir /media/horizon/Database/robotic_datasets/visual_learning --dataset_name mnist --max_epochs 100 --pretrained_model_path ./output/final_model.pth
python3 ./src/train_model.py --model cnn --dataset_dir /media/horizon/Database/robotic_datasets/visual_learning --dataset_name mnist --max_epochs 100 --pretrained_model_path ./output/final_model.pth
python3 ./src/train_model.py --model resnet --dataset_dir /media/horizon/Database/robotic_datasets/visual_learning --dataset_name mnist --max_epochs 100 --pretrained_model_path ./output/final_model.pth
python3 ./src/train_model.py --model vit --dataset_dir /media/horizon/Database/robotic_datasets/visual_learning --dataset_name mnist --max_epochs 100 --pretrained_model_path ./output/final_model.pth
