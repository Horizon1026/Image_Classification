#ifndef _MNIST_RESNET_H_
#define _MNIST_RESNET_H_

#include "basic_type.h"
#include "slam_log_reporter.h"
#include "memory"
#include "libtorch.h"

namespace NN_SLAM {

/* Class MnistResNet Declaration. */
class MnistResNet {

public:
    // Input size: patch_size x N x M x in_planes.
    // Output size: patch_size x N x M x planes.
    struct ResNetBlock : torch::nn::Module {
        ResNetBlock(int32_t in_planes, int32_t planes, int32_t stride = 1) :
            conv1(torch::nn::Conv2dOptions(in_planes, planes, 3).stride(stride).padding(1).bias(false)),
            bn1(planes),
            conv2(torch::nn::Conv2dOptions(planes, planes, 3).stride(1).padding(1).bias(false)),
            bn2(planes) {
            register_module("conv1", conv1);
            register_module("bn1", bn1);
            register_module("conv2", conv2);
            register_module("bn2", bn2);

            if (stride != 1 || in_planes != planes) {
                shortcut = torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_planes, planes, 1).stride(stride).bias(false)),
                    torch::nn::BatchNorm2d(planes)
                );
            } else {
                shortcut = torch::nn::Sequential();
            }
            register_module("shortcut", shortcut);
        }

        torch::Tensor forward(torch::Tensor x) {
            auto out = conv1->forward(x);
            out = bn1->forward(out);
            out = torch::relu(out);

            out = conv2->forward(out);
            out = bn2->forward(out);

            if (!shortcut->is_empty()) {
                out += shortcut->forward(x);
            } else {
                out += x;
            }
            out = torch::relu(out);
            return out;
        }

        torch::nn::Conv2d conv1{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr};
        torch::nn::Conv2d conv2{nullptr};
        torch::nn::BatchNorm2d bn2{nullptr};
        torch::nn::Sequential shortcut{nullptr};
    };

    struct ResNet : torch::nn::Module {
        ResNet(int32_t num_classes = 10, int32_t init_channels = 64) :
            fc(init_channels * 4, num_classes) {
            layer1 = torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(/*in_channels=*/1, /*out_channels=*/init_channels, /*kernel_size=*/3)
                    .stride(1).padding(1).bias(false)),
                torch::nn::BatchNorm2d(init_channels),
                torch::nn::ReLU(),
                // Parameters of MaxPool2d: kernel_size, stride, padding.
                // If stride is not specified, it is set to kernel_size.
                torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(3).stride(2).padding(1))
            );
            layer2 = make_layer(init_channels, init_channels, 2, 1);
            layer3 = make_layer(init_channels, init_channels * 2, 2, 2);
            layer4 = make_layer(init_channels * 2, init_channels * 4, 2, 2);

            register_module("layer1", layer1);
            register_module("layer2", layer2);
            register_module("layer3", layer3);
            register_module("layer4", layer4);
            register_module("fc", fc);
        }

        torch::nn::Sequential make_layer(int32_t in_planes, int32_t planes, int32_t num_blocks, int32_t stride) {
            torch::nn::Sequential layers;
            layers->push_back(ResNetBlock(in_planes, planes, stride));
            for (int i = 1; i < num_blocks; i++) {
                layers->push_back(ResNetBlock(planes, planes));
            }
            return layers;
        }

        torch::Tensor forward(torch::Tensor x) {
            /*  x.sizes() = [64, 1, 28, 28]
                after layer1: x.sizes() = [64, 64, 14, 14]
                after layer2: x.sizes() = [64, 64, 14, 14]
                after layer3: x.sizes() = [64, 128, 7, 7]
                after layer4: x.sizes() = [64, 256, 4, 4]
                after avg_pool2d: x.sizes() = [64, 256, 1, 1]
                after view: x.sizes() = [64, 256]
                after fc: x.sizes() = [64, 10]
            */
            x = layer1->forward(x);
            x = layer2->forward(x);
            x = layer3->forward(x);
            x = layer4->forward(x);
            x = torch::avg_pool2d(x, 4);
            x = x.view({x.size(0), -1});
            x = fc->forward(x);
            x = torch::log_softmax(x, 1);
            return x;
        }

        torch::nn::Sequential layer1;
        torch::nn::Sequential layer2;
        torch::nn::Sequential layer3;
        torch::nn::Sequential layer4;
        torch::nn::Linear fc;
    };

public:
    MnistResNet() = default;
    virtual ~MnistResNet() = default;

    void Train(const std::string &mnist_data_path);
    void Test(const std::string &mnist_data_path);

private:
    std::shared_ptr<ResNet> model_ = std::make_shared<ResNet>(10, 32);
    struct Options {
        std::string output_file = "../output/mnist_resnet.pt";
        int32_t max_epoch = 8;
        int32_t batch_size = 64;
    } options_;

};

}

#endif // end of _MNIST_RESNET_H_
