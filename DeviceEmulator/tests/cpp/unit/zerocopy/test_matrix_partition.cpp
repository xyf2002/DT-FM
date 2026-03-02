#include "backend/server_base.h"

int main() {
  // std::vector<int> list = {2, 3, 4};
  // auto result = CartesianProduct(list);
  // for (auto& res : result) {
  //   for (auto& elem : res) {
  //     std::cout << elem << " ";
  //   }
  //   std::cout << std::endl;
  // }

  torch::Tensor mat_a =
      torch::randn({2, 3, 4}).contiguous().to(torch::kFloat32);
  torch::Tensor mat_b = torch::randn({3, 4}).contiguous().to(torch::kFloat32);

  auto ref = torch::matmul(mat_a, mat_b.transpose(0, 1));

  auto output = CreateOutputMatrix(mat_a, mat_b);

  std::cout << "mat_a " << mat_a << std::endl;
  std::cout << "mat_b " << mat_b << std::endl;
  std::cout << "output shape " << output.sizes() << std::endl;

  auto partitions = PartitionMatrices(mat_a, mat_b, 2);
  // assert(partitions.size() == 6);
  for (auto& partition : partitions) {
    fprintf(stderr, "row: %ld, col: %ld, pivot: %ld, h_dim: %ld\n",
            partition->row, partition->col, partition->pivot, partition->h_dim);
    // create tensor from partition
    auto [r_ptr, r_size] = partition->mat[0];
    auto [c_ptr, c_size] = partition->mat[1];

    int64_t row_size = r_size / partition->h_dim / sizeof(float);
    int64_t col_size = c_size / partition->h_dim / sizeof(float);

    assert(row_size * partition->h_dim * sizeof(float) == r_size);
    assert(col_size * partition->h_dim * sizeof(float) == c_size);

    fprintf(stderr, "row_size: %ld, col_size: %ld\n", row_size, col_size);
    fprintf(stderr, "r_ptr: %p, c_ptr: %p\n", r_ptr, c_ptr);
    torch::TensorOptions options = torch::TensorOptions()
                                       .dtype(torch::kFloat32)
                                       .device(torch::kCPU)
                                       .layout(torch::kStrided)
                                       .requires_grad(false);
    void* r_ptr_cpy = malloc(r_size);
    void* c_ptr_cpy = malloc(c_size);

    memcpy(r_ptr_cpy, r_ptr, r_size);
    memcpy(c_ptr_cpy, c_ptr, c_size);

    auto row =
        torch::from_blob(r_ptr_cpy, {row_size, partition->h_dim}, options);
    auto col =
        torch::from_blob(c_ptr_cpy, {col_size, partition->h_dim}, options);

    std::cout << "row " << row << std::endl;
    std::cout << "col " << col << std::endl;

    auto result = torch::mm(row, col.transpose(0, 1));
    std::cout << "result " << result << std::endl;

    UpdateMatrixBlock(output, result, partition->row, partition->col,
                      partition->pivot, 2);
  }

  // output should be same as ref
  std::cout << "output " << output << std::endl;
  std::cout << "ref " << ref << std::endl;
  assert(torch::allclose(output, ref));

  return 0;
}
