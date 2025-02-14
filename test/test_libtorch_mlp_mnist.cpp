#include "basic_type.h"
#include "slam_log_reporter.h"
#include "mnist_mlp.h"

#include "enable_stack_backward.h"

int main(int argc, char **argv) {
    ReportColorInfo(">> Test model mlp on mnist.");
    ReportColorInfo("torch::cuda::is_available() = " << torch::cuda::is_available());
    ReportColorInfo("torch::cuda::cudnn_is_available() = " << torch::cuda::cudnn_is_available());
    ReportColorInfo("torch::cuda::device_count() = " << torch::cuda::device_count());

    NN_SLAM::MnistMlp solver;
    solver.Train("/mnt/d/My_Github/Datasets/MNIST/raw");
    solver.Test("/mnt/d/My_Github/Datasets/MNIST/raw");

    return 0;
}
