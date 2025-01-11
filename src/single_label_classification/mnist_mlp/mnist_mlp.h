#ifndef _MNIST_MLP_H_
#define _MNIST_MLP_H_

#include "basic_type.h"
#include "memory"
#include "libtorch.h"

namespace NN_SLAM {

/* Class MnistMlp Declaration. */
class MnistMlp {

public:
    struct Model : torch::nn::Module {
        Model() :
            fc1(28 * 28, 512),
            fc2(512, 10) {
            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        torch::Tensor forward(torch::Tensor x) {
            x = x.view({x.size(0), -1});
            x = torch::relu(fc1->forward(x));
            x = fc2->forward(x);
            x = torch::log_softmax(x, 1);
            return x;
        }

        torch::nn::Linear fc1{nullptr};
        torch::nn::Linear fc2{nullptr};
    };

public:
    MnistMlp() = default;
    virtual ~MnistMlp() = default;

    void Train(const std::string &mnist_data_path);
    void Test(const std::string &mnist_data_path);

private:
    std::shared_ptr<Model> model_ = std::make_shared<Model>();
    struct Options {
        std::string output_file = "../output/mnist_mlp.pt";
        int32_t max_epoch = 5;
        int32_t batch_size = 64;
    } options_;

};

}

#endif // end of _MNIST_MLP_H_
