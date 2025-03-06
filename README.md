# Image_Classification
Some image classifier.

# Components
- [x] Mnist - MLP.
- [x] Mnist - CNN.
- [x] Mnist - ResNet.
- [x] Mnist - ViT.

# Dependence (C++)
- libtorch (in Slam_Utility/3rd_libraries)
- Slam_Utility

# Dependence (python)
- pytorch

# Compile and Run
- 第三方仓库的话需要自行 apt-get install 安装
- 拉取 Dependence 中的源码，在当前 repo 中创建 build 文件夹，执行标准 cmake 过程即可
```bash
mkdir build
cmake ..
make -j
```
- 编译成功的可执行文件就在 build 中，具体有哪些可执行文件可参考 run.sh 中的列举。可以直接运行 run.sh 来依次执行所有可执行文件

```bash
sh run.sh
```

# Tips
- 欢迎一起交流学习，不同意商用；
- 部分有 C++ 和 python 两个版本。初期有尝试都用 libtorch 做训练和直接部署，但遇到了不少兼容性的问题，试错成本也因为编译速度导致较大，因此还是回归了 pytorch 做训练，部署的时候再考虑 libtorch.
