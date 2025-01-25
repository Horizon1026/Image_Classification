# 如果没有发现 ../output 路径，则创建一个
if [ ! -d "./output" ]; then
    mkdir output
fi

cd ./build
./test_libtorch_mlp_mnist
./test_libtorch_cnn_mnist
./test_libtorch_resnet_mnist
cd ..