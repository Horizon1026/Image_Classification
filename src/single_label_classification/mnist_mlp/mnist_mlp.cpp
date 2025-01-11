#include "mnist_mlp.h"
#include "slam_log_reporter.h"

namespace NN_SLAM {

void MnistMlp::Train(const std::string &mnist_data_path) {
    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST(mnist_data_path.c_str()).map(
            torch::data::transforms::Stack<>()), options_.batch_size);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model_->to(device);
    torch::optim::SGD optimizor(model_->parameters(), 0.01);

    for (int32_t epoch = 0; epoch < options_.max_epoch; ++epoch) {
        int32_t batch_index = 0;

        for (auto &batch : *data_loader) {
            // Move data and target to selected device.
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            // Reset gradients.
            optimizor.zero_grad();
            // Execute the model on the input data.
            torch::Tensor prediction = model_->forward(data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = torch::nll_loss(prediction, target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizor.step();
            // Print the loss and check point.
            if (++batch_index % 100 == 0) {
                ReportInfo("[Train] epoch [" << epoch << "], batch_idx [" << batch_index << "], loss [" << loss.item<float>() << "].");
                torch::save(model_, options_.output_file.c_str());
            }
        }
    }
}

void MnistMlp::Test(const std::string &mnist_data_path) {
    auto dataset = torch::data::datasets::MNIST(mnist_data_path.c_str(),
        torch::data::datasets::MNIST::Mode::kTest).map(
            torch::data::transforms::Stack<>());
    const auto dataset_size = dataset.size().value();
    auto data_loader = torch::data::make_data_loader(std::move(dataset), options_.batch_size);

    torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
    model_->to(device);

    float average_loss = 0.0;
    int32_t correct_cnt = 0;
    for (auto &batch : *data_loader) {
        // Move data and target to selected device.
        auto data = batch.data.to(device);
        auto target = batch.target.to(device);
        // Execute the model on the input data.
        torch::Tensor output = model_->forward(data);
        // Compute a loss value to judge the output of our model.
        torch::Tensor loss = torch::nll_loss(output, target, /*weight=*/{}, torch::Reduction::Sum);
        average_loss += loss.item<float>();
        // Check correction.
        torch::Tensor prediction = output.argmax(1);
        correct_cnt += prediction.eq(target).sum().item<int32_t>();
    }

    average_loss /= static_cast<float>(dataset_size);
    const float corr_rate = static_cast<float>(correct_cnt) / static_cast<float>(dataset_size);

    ReportInfo("[Test] average loss [" << average_loss << "], correct rate [" << corr_rate <<
        "], correct cnt [" << correct_cnt << "/" << dataset_size << "]");
}

}
