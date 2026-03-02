/*
 * slice.h

 *
 *  Created on: 2018年1月15日
 *      Author: sing.peng
 */

#ifndef UEVENT_BASE_SLICE_H_
#define UEVENT_BASE_SLICE_H_

#include <memory>

namespace base {

class Slice {
 public:
  Slice(uint64_t size);
  Slice(void* data_, uint64_t size, bool data_mem_new = true);
  ~Slice();

  void* data() const { return data_; }

  void set_data(void* data) { data_ = data; }

  uint64_t size() const { return size_; }

  void set_size(uint64_t size) { size_ = size; }

 private:
  void* data_;
  uint64_t size_;
  bool data_mem_new_;
};

typedef std::shared_ptr<Slice> SlicePtr;

} /* namespace base */

#endif /* UEVENT_BASE_SLICE_H_ */
