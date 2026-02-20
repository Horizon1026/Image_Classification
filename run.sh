python3 ./src/train_model.py --model mlp --dataset_name mnist --max_epochs 10 --pretrained_model_path ./output/final_model.pth
python3 ./src/train_model.py --model cnn --dataset_name mnist --max_epochs 10 --pretrained_model_path ./output/final_model.pth
python3 ./src/train_model.py --model resnet --dataset_name mnist --max_epochs 10 --pretrained_model_path ./output/final_model.pth
python3 ./src/train_model.py --model vit --dataset_name mnist --max_epochs 10 --pretrained_model_path ./output/final_model.pth
