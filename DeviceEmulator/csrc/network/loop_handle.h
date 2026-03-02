#ifndef UEVENT_LOOP_HANDLE_H_
#define UEVENT_LOOP_HANDLE_H_
#include <assert.h>

#include <atomic>

#include "uevent.h"

namespace uevent {

class LoopHandle {
 public:
  LoopHandle() : weight_(0), refs_(0) {}

  uevent::UeventLoop* GetLoop() { return loop_; }

  void SetLoop(uevent::UeventLoop* loop) {
    loop_ = loop;
    return;
  }
  inline uint64_t GetWeight() const { return weight_.load(); }

  inline void SetWeight(uint64_t w) { weight_.store(w); }

  inline int32_t GetRefs() const { return refs_.load(); }

  inline void IncRefs() { refs_.fetch_add(1); }

  inline void DecRefs() {
    refs_.fetch_sub(1);
    assert(refs_.load() >= 0);
  }

 protected:
  uevent::UeventLoop* loop_;

 private:
  std::atomic<uint64_t> weight_;
  std::atomic<int32_t> refs_;
};

}  // namespace uevent

#endif
