#include <torch/extension.h>

#include "intercept/interceptor.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Removed MemoryManagerClient bindings since gRPC client was removed

  m.def("sgemm_", &sgemm_, "sgemm_");
}
