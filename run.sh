# 如果没有发现 ../output 路径，则创建一个
if [ ! -d "./output" ]; then
    mkdir output
fi

python3 ./src/train_mlp.py
python3 ./src/train_cnn.py
python3 ./src/train_resnet.py
python3 ./src/train_vit.py

cd ./build
./test_libtorch_mlp_mnist
./test_libtorch_cnn_mnist
./test_libtorch_resnet_mnist
cd ..