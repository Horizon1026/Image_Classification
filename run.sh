# 如果没有发现 ../output 路径，则创建一个
if [ ! -d "./output" ]; then
    mkdir output
fi

python3 ./src/mnist_mlp/train_and_test.py
python3 ./src/mnist_cnn/train_and_test.py
python3 ./src/mnist_resnet/train_and_test.py
python3 ./src/mnist_vit/train_and_test.py
