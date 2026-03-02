/*
 * slice.cpp
 *
 *  Created on: 2018年1月15日
 *      Author: sing.peng
 */

#include "slice.h"

#include <stdlib.h>

namespace base {

Slice::Slice(uint64_t size) : data_(NULL), size_(0), data_mem_new_(true) {
  // TODO Auto-generated constructor stub
  data_ = new char[size];
}

Slice::Slice(void* data_, uint64_t size, bool data_mem_new)
    : data_(data_), size_(size), data_mem_new_(data_mem_new) {
  // TODO Auto-generated constructor stub
}

Slice::~Slice() {
  // TODO Auto-generated destructor stub
  if (data_mem_new_)
    delete[]((char*)data_mem_new_);
  else
    free(data_);
}

} /* namespace base */
