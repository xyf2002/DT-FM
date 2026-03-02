#include <torch/torch.h>

int main() {
  // create torch 2x2 tensor
  auto tensor = torch::randn({2, 2}).contiguous();
  auto tensor_weight = torch::randn({2, 2}).contiguous();

  // print tensor
  std::cout << "tensor " << tensor << std::endl;
  std::cout << "tensor_weight " << tensor << std::endl;

  // // get float data_ptr

  // for (int i = 0; i < tensor.numel(); i++) {
  //   std::cout << "data_ptr "  << data_ptr[i] << std::endl;
  // }

  std::string serialized;
  void* data_ptr = tensor.data_ptr();
  serialized.append(reinterpret_cast<char*>(data_ptr),
                    tensor.numel() * sizeof(float));
  auto options = tensor.options();

  // deserialize
  torch::Tensor new_tensor =
      torch::from_blob(serialized.data(), tensor.sizes(), options);

  // print new tensor
  std::cout << "new_tensor " << new_tensor << std::endl;

  std::string serialized_weight;
  data_ptr = tensor_weight.data_ptr();
  serialized_weight.append(reinterpret_cast<char*>(data_ptr),
                           tensor_weight.numel() * sizeof(float));
  auto options_weight = tensor_weight.options();

  // deserialize
  torch::Tensor new_tensor_weight = torch::from_blob(
      serialized_weight.data(), tensor_weight.sizes(), options_weight);

  // print new tensor
  std::cout << "new_tensor_weight " << new_tensor_weight << std::endl;

  auto gpu_new_tensor = new_tensor.to(torch::kCUDA);
  auto gpu_new_tensor_weight = new_tensor_weight.to(torch::kCUDA);

  // perform matrix multiplication
  auto result = torch::matmul(gpu_new_tensor, gpu_new_tensor_weight);

  auto cpu_result = result.to(torch::kCPU);

  // print result
  std::cout << "cpu_result " << cpu_result << std::endl;
}
