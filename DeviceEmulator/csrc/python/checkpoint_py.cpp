#include <torch/extension.h>

#include "checkpoint/archer_tensor_handle.h"
#include "checkpoint/checkpoint_handle.h"
#include "memory/shared_memory.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ArcherTensorHandle>(m, "ArcherTensorHandle")
      .def(py::init<const std::string&>())
      .def("is_tensor_index_initialized",
           &ArcherTensorHandle::IsTensorIndexInitialized,
           "Check if tensor index is initialized")
      .def("offload_tensor", &ArcherTensorHandle::OffloadTensor,
           "Offload tensor to disk")
      .def("is_tensor_offloaded", &ArcherTensorHandle::IsTensorOffloaded,
           "Check if tensor is offloaded");
  m.def(
      "set_tensor_shm",
      [](torch::Tensor& tensor, std::string& name, size_t size) {
        void* ptr = OpenSharedMemory(name.c_str(), size);
        // deleter does nothing since memory is managed by emulator shared
        // memory
        // check if ptr bytes are all zeros
        //    bool is_zero = true;
        //    for (size_t i = 0; i < size; i++) {
        //      if (((char*)ptr)[i] != 0) {
        //        is_zero = false;
        //        break;
        //      }
        //    }
        //    LOG_FATAL_IF(
        //        is_zero,
        //        "Read all zeros, file: {}, offset: {}, size: {}, aligned_size:
        //        {}", name.c_str(), 0, size, size);
        ShmMeta meta{
            .id = -1,
            .ptr = ptr,
            .size = size,
            .name = name,
            .is_remote = true,
        };
        kCachingAllocator->InsertShmMeta(meta);
        LOG_DEBUG << "set_tensor_shm: name: " << name << ", size: " << size
                  << ", ptr: " << ptr;
        tensor.set_data(torch::from_blob(ptr, tensor.sizes(), tensor.strides(),
                                         DoNothingDeleter<void>{},
                                         tensor.options()));
      },
      "Set tensor shared memory");

  // Removed MemoryManagerClient bindings since gRPC client was removed
}
