#ifndef _MNIST_CNN_H_
#define _MNIST_CNN_H_

#include "basic_type.h"
#include "memory"
#include "libtorch.h"

namespace NN_SLAM {

/* Class MnistCnn Declaration. */
class MnistCnn {

public:
    struct Model : torch::nn::Module {
        Model() :
            conv1(torch::nn::Conv2dOptions(1, 2, /*kernel_size=*/5)),
            conv2(torch::nn::Conv2dOptions(2, 4, /*kernel_size=*/3)),
            fc1(484, 128),
            fc2(128, 10) {
            register_module("conv1", conv1);
            register_module("conv2", conv2);
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        torch::Tensor forward(torch::Tensor x) {
            // input : 1*28*28.
            // conv1 : 28 - 5 + 1 = 24.
            x = torch::relu(conv1->forward(x));
            // input : 24*24*2.
            // conv2 : 24 - 3 + 1 = 22.
            // max_pool : 11 * 11 * 4 = 484.
            x = torch::max_pool2d(torch::relu(conv2->forward(x)), 2);
            x = x.view({-1, 484});
            // w : 128 * 484.
            x = torch::relu(fc1->forward(x));
            // w : 10 * 128.
            x = fc2->forward(x);
            x = torch::log_softmax(x, 1);
            return x;
        }

        torch::nn::Conv2d conv1;
        torch::nn::Conv2d conv2;
        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
    };

public:
    MnistCnn() = default;
    virtual ~MnistCnn() = default;

    void Train(const std::string &mnist_data_path);
    void Test(const std::string &mnist_data_path);

private:
    std::shared_ptr<Model> model_ = std::make_shared<Model>();
    std::string output_file_ = "../output/mnist_cnn.pt";

};

}

#endif // end of _MNIST_CNN_H_
