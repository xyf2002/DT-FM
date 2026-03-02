#include <torch/torch.h>

#include <chrono>
#include <iostream>

int main(int argc, char* argv[]) {
  // Parameters
  const int64_t rows = 8192;
  const int64_t cols = 8192;

  // // read a number of arguments
  // if (argc > 1) {
  //   rows = std::stoi(argv[1]);
  //   cols = std::stoi(argv[1]);
  // }
  // if (argc > 2) {
  //   cols = std::stoi(argv[2]);
  // }

  {
    // Create a large tensor
    auto large_tensor_col = torch::rand({rows, cols});

    // Measure column-wise copy
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < cols; ++i) {
      auto col_copy = large_tensor_col.index({torch::indexing::Slice(), i});
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto col_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time)
                            .count();
    std::cout << "Time taken for column-wise copy: " << col_duration
              << " microseconds." << std::endl;
  }

  {
    auto large_tensor_row = torch::rand({rows, cols});
    // Measure row-wise copy
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int64_t i = 0; i < rows; ++i) {
      auto row_copy = large_tensor_row[i];
    }
    auto end_time = std::chrono::high_resolution_clock::now();
    auto row_duration = std::chrono::duration_cast<std::chrono::microseconds>(
                            end_time - start_time)
                            .count();
    std::cout << "Time taken for row-wise copy: " << row_duration
              << " microseconds." << std::endl;
  }

  return 0;
}
