#pragma once

#include <algorithm>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/env_cfg.h"
#include "common/pytorch_defs.h"
#include "device_tracker.h"
#include "morphling.pb.h"
#include "network/uevent.h"
#include "network/ueventloop_thread.h"
#include "sched_policy.h"
#include "server_base.h"

namespace morphling {
namespace backend {

struct BatchSendRecv {
  size_t sent;
  size_t recv;
  bool ignore;

  BatchSendRecv() : sent(0), recv(0), ignore(false) {}
  void Reset() {
    sent = 0;
    recv = 0;
    ignore = false;
  }
  void IncSent() { ++sent; }
  void IncRecv() { ++recv; }
  void SetIgnore() {
    ignore = true;
    ++sent;
  }
  bool Complete() { return sent == recv; }
  bool HasIgnore() { return ignore; }
};

class ProxySvrHandle : public uevent::LoopHandle {
 public:
  ProxySvrHandle(ProxyEnvCfg& ctx, uevent::UeventLoop* loop);

  static void ThreadInit(uevent::UeventLoop* loop);
  static uevent::LoopHandle* CreateMyself(ProxyEnvCfg& ctx,
                                          uevent::UeventLoop* loop) {
    return new ProxySvrHandle(ctx, loop);
  }

 public:
  void RequestCb(const uevent::ConnectionUeventPtr& conn);
  void RequestWriteCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);
  void SendInLoop(const uevent::ConnectionUeventPtr& conn,
                  const MatrixPartitionPtr partition);

  // Redistribute idle partitions to devices
  void SendIdlePartitions();

  // send perf request to device
  void SendPerfInLoop(const uevent::ConnectionUeventPtr& conn,
                      const DevicePerfPtr perf);

  // Device registration methods
  void SendRegisterRequest(const uevent::ConnectionUeventPtr& conn);
  MessageHandlerSignature HandleRegisterResponse;

 private:
  // Message decoding and dispatching
  void DecodeAndDispatch(const uevent::ConnectionUeventPtr& conn,
                         const void* payload, size_t size);

  // Message handlers following MessageHandler interface
  MessageHandlerSignature HandleMatMul;
  MessageHandlerSignature HandleDevicePerf;

 private:
  ProxyEnvCfg& ctx_;
  uevent::UeventLoop* loop_;
  std::unordered_map<std::string, uint32_t> conn_inflight_;
  std::deque<std::function<void()>> task_queue_;

  // Handshake tracking: maps connection address to handshake state
  // true = handshake complete, false = waiting for device_id
  std::unordered_map<std::string, bool> conn_handshake_complete_;
  std::mutex handshake_mutex_;

  // std::unordered_map<std::string, DeviceProfileData> device_info_;
  // Scheduling policy is now in ctx_.sched_policy
};

class ProxySvrImpl : public std::enable_shared_from_this<ProxySvrImpl> {
 public:
  ProxySvrImpl(ProxyEnvCfg& context);
  ~ProxySvrImpl();
  void Initialize(uevent::UeventLoop* loop);

  void DispatchMatMulAsync(torch::Tensor& mat_a, torch::Tensor& mat_b);
  torch::Tensor WaitMatMul(int oid);
  torch::Tensor& GetOutputMatrix(int oid) { return outputs_[oid]; }

  void IncRspCbCount(int oid, size_t count);
  void DecRspCbCount(int oid, size_t count) { rsp_cb_counts_[oid] += count; }

  // Connection and device queries - delegates to DevicePartitionTracker
  size_t GetConnectionCount() const {
    return DEVICE_TRACKER.GetConnectedDeviceCount();
  }
  size_t GetRegisteredDeviceCount() const {
    return DEVICE_TRACKER.GetConnectedDeviceCount();
  }
  bool IsDeviceRegistered(const std::string& addr) const {
    return registered_devices_.find(addr) != registered_devices_.end();
  }
  void RegisterDevice(const std::string& addr, const DeviceProfileData& info) {
    registered_devices_[addr] = info;
  }
  void UnregisterDevice(const std::string& addr) {
    registered_devices_.erase(addr);
  }

  // Device failure handling
  void HandleDeviceFailure(int64_t failed_device_id);
  void CheckFailedPartitions();  // Periodic check for failed partitions

 private:
  void ConnectionSuccessCb(const uevent::ConnectionUeventPtr& conn);
  void ConnectionClosedCb(const uevent::ConnectionUeventPtr& conn);
  void RequestCb(const uevent::ConnectionUeventPtr& conn);
  void RequestWriteCb(const uevent::ConnectionUeventPtr& conn);

  // Rephrase partitions with optional excluded devices for retry scenarios
  void RephrasePartitions(
      std::vector<MatrixPartitionPtr>& partitions,
      const std::unordered_set<int64_t>& excluded_devices = {});

 private:
  ProxyEnvCfg& ctx_;
  uevent::UeventLoop* loop_;
  std::shared_ptr<uevent::ListenerUevent> listener_;

  // std::unordered_map<std::string, uevent::ConnectionUeventPtr> conn_map_;
  // Note: Device connections by device_id are now managed by
  // DevicePartitionTracker Access via
  // DEVICE_TRACKER.SetDeviceConnection/GetDeviceConnection/RemoveDeviceConnection
  // Note: Device ID management is now handled by DevicePartitionTracker
  // singleton

  std::atomic_int mm_count_{0};
  std::atomic_int gemm_id_count_{0};  // global gemm operation id counter
  std::vector<torch::Tensor> outputs_;
  std::vector<std::atomic_ullong> rsp_cb_counts_;
  std::vector<std::unordered_set<TensorKey>> device_tensors_;

  // Note: Partition tracking is now handled by DevicePartitionTracker singleton
  // Access via DEVICE_TRACKER macro

  std::unordered_map<std::string, DeviceProfileData> registered_devices_;
  uevent::TimerId
      idle_partition_redistribute_timer_;  // Timer for periodic idle partition
                                           // redistribution
  uevent::TimerId failed_partition_check_timer_;  // Timer for periodic
                                                  // partition health checks
};

typedef std::shared_ptr<ProxySvrImpl> ProxySvrImplPtr;

class ProxySvr {
 public:
  class Status {
   public:
    enum StatusType {
      kOK = 0,
      kFatal,
      kUnknown,
    };

   public:
    Status() : status_(kOK), err_() {}
    const std::string& err() { return err_; }
    bool OK() { return status_ == kOK; }
    void SetError(StatusType status, const std::string& err) {
      status_ = status;
      if (status_str_.find(status) == status_str_.end()) status_ = kUnknown;
      err_ = status_str_.at(status_) + err_;
    }

   private:
    StatusType status_;
    std::string err_;
    static const std::map<StatusType, std::string> status_str_;
  };

 public:
  ProxySvr();
  void Initialize(const std::string& cfg_file);
  void Start();
  void SetCacheEnabled(bool enabled) {
    context_.enable_cli_cache = enabled ? 1 : 0;
  }
  void DispatchMatMulAsync(torch::Tensor& mat_a, torch::Tensor& mat_b) {
    svr_->DispatchMatMulAsync(mat_a, mat_b);
  }
  torch::Tensor WaitMatMul(int oid) { return svr_->WaitMatMul(oid); }
  size_t GetConnectionCount() const { return svr_->GetConnectionCount(); }
  size_t GetRegisteredDeviceCount() const {
    return svr_->GetRegisteredDeviceCount();
  }

 private:
  ProxySvrImplPtr svr_;
  ProxyEnvCfg context_;
  std::shared_ptr<uevent::UeventLoopThread> loop_thread_;
};

}  // namespace backend
}  // namespace morphling
