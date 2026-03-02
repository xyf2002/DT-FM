#include "proxy_svr.h"

#include <chrono>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/stats.h"
#include "device_tracker.h"
#include "network/eventloop_libevent.h"
#include "network/listener_libevent.h"
#include "partition_tracker.h"
#include "proto_base.h"
#include "server_base.h"
#include "utils/logger.h"
#include "utils/thread_affinity.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <atomic>
#include <iostream>
#include <set>

namespace morphling {
namespace backend {

// Global atomic counter to assign each thread to a different CPU core
static std::atomic<int> g_thread_core_counter(0);

static void PinThreadToNextAvailableCore() {
  int num_cpus = morphling::GetOnlineCoreCount();
  int core_id =
      g_thread_core_counter.fetch_add(1, std::memory_order_relaxed) %
      num_cpus;
  morphling::PinThreadToCore(core_id);
}

/*********************************ProxySvrHandle***********************************/

ProxySvrHandle::ProxySvrHandle(ProxyEnvCfg& ctx, UeventLoop* loop)
    : ctx_(ctx), loop_(loop) {
  SRV_STATS->Initialize();
  // Scheduling policy is now in ctx_.sched_policy
}

void ProxySvrHandle::ThreadInit(uevent::UeventLoop* loop) {
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);

  // Pin this thread to the next available CPU core in round-robin fashion
  // This is called in the worker thread context to ensure correct thread
  // binding
  PinThreadToNextAvailableCore();
}

// Helper for zero-copy send cleanup of SerializationBuffer (registration msgs)
static void SerializationBufferSendCleanup(const void* /*data*/, size_t /*len*/,
                                           void* arg) {
  delete static_cast<SerializationBufferPtr*>(arg);
}

void ProxySvrHandle::RequestWriteCb(const uevent::ConnectionUeventPtr& conn) {
  size_t readable = conn->ReadableLength();
  LOG_DEBUG << "RequestWriteCb readable: " << readable;
}

void ProxySvrHandle::SendRegisterRequest(const ConnectionUeventPtr& conn) {
  LOG_INFO << "Sending registration request to "
           << conn->GetPeerAddress().ToString();

  DeviceRegisterRequest request;
  auto buffer = request.Serialize();

  LOG_DEBUG << "Raw registration request data (hex): "
            << BinaryToHex(static_cast<const uint8_t*>(buffer->GetBuffer()),
                           buffer->GetSize());

  // Zero-copy send: buffer ref-count prevents deallocation until libevent done
  auto* ref = new SerializationBufferPtr(buffer);
  int ret = conn->SendDataZeroCopy(buffer->GetBuffer(), buffer->GetSize(),
                                   SerializationBufferSendCleanup, ref);
  if (ret < 0) {
    LOG_ERROR << "Failed to send registration request";
    conn->ForceClose();
    return;
  }
}

void ProxySvrHandle::RequestCb(const ConnectionUeventPtr& conn) {
  while (true) {
    size_t readable = conn->ReadableLength();

    int ret = 0;
    uint32_t packsize = 0;
    ret = conn->ReceiveData(&packsize, sizeof(uint32_t));
    if (ret < 0) {
      LOG_ERROR << "ReceiveData packsize err";
      return;
    }
    packsize = ntohl(packsize);
    size_t datasize = packsize + sizeof(packsize);

    LOG_TRACE << "packsize: " << packsize << ", datasize: " << datasize
              << ", readable: " << readable;

    if (readable < datasize) {
      return;
    }

    // Zero-copy receive: get contiguous pointer into evbuffer
    unsigned char* raw_data = conn->PullupData(datasize);
    if (raw_data == nullptr) {
      LOG_ERROR << "PullupData failed for size " << datasize;
      return;
    }

    // Decode and dispatch message (processes data in-place)
    DecodeAndDispatch(conn, raw_data, datasize);

    // Drain after processing is complete
    ret = conn->DrainData(datasize);
    if (ret < 0) {
      LOG_ERROR << "DrainData err";
      return;
    }
  }
}

void ProxySvrHandle::DecodeAndDispatch(const ConnectionUeventPtr& conn,
                                       const void* payload, size_t size) {
  // Step 1: Decode proto message header to get message type
  int32_t message_type = GetMessageType(payload, size);

  if (message_type < 0) {
    LOG_ERROR << "Failed to decode message type";
    return;
  }

  // Step 2: Dispatch to appropriate handler based on message type
  string client_addr = conn->GetPeerAddress().ToString();

  switch (message_type) {
    case morphling::global_api::DEVICE_PROFILE_DATA:
      HandleRegisterResponse(conn, payload, size);
      break;

    case morphling::global_api::COMPUTE_GEMM_DATA:
      // Check if client is connected via tracker
      {
        auto& tracker = DEVICE_TRACKER;
        int64_t device_id = tracker.GetDeviceIdByAddr(client_addr);
        if (device_id == -1 || !tracker.IsDeviceConnected(device_id)) {
          LOG_ERROR << "Client " << client_addr
                    << " not registered or not connected, disconnecting";
          conn->ForceClose();
          return;
        }
      }
      HandleMatMul(conn, payload, size);
      conn_inflight_[client_addr] -= 1;

      if (!task_queue_.empty()) {
        auto task = task_queue_.front();
        task_queue_.pop_front();
        task();
      }
      break;

    default:
      LOG_ERROR << "Unknown message type: " << message_type;
      break;
  }
}

void ProxySvrHandle::HandleRegisterResponse(const ConnectionUeventPtr& conn,
                                            const void* payload, size_t size) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_DEBUG << "Received device profile data from " << client_addr
            << ", size=" << size << ", hex: "
            << BinaryToHex(static_cast<const uint8_t*>(payload), size) << "";

  // Use standard Deserialize interface
  DeviceProfileData profile;
  profile.Deserialize(payload, size);

  // Register device in tracker
  int64_t device_id = DEVICE_TRACKER.RegisterDevice(client_addr, profile);

  // Store connection in tracker
  DEVICE_TRACKER.SetDeviceConnection(device_id, conn);

  // Store device info
  // device_info_[client_addr] = profile;

  LOG_DEBUG << "Client " << client_addr
            << " registered with device_id=" << device_id << ": "
            << profile.DebugString();

  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, this));
}

void ProxySvrHandle::HandleMatMul(const ConnectionUeventPtr& conn,
                                  const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();

  // Record RECEIVE/DOWNLOAD start time (virtual time)
  uint64_t vt_receive_start = VirtualClockNow();

  // Use standard Deserialize interface
  MatrixPartition partition;
  partition.Deserialize(payload, size);

  // Log RECEIVE START after getting device_id from partition
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "RECEIVE", "START", vt_receive_start,
                                     vt_receive_start);

  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << part_key << " RSP Deserialization time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  // Record bytes received (download request from device)
  DEVICE_TRACKER.RecordBytesReceived(partition.dev_id, size);

  // Log download throughput after receiving response
  double download_tp = DEVICE_TRACKER.GetDownloadThroughput(partition.dev_id);
  double last_packet_tp =
      DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp =
      DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();

  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us,
                                              end_us);

  LOG_INFO << "[HandleMatMul] Device " << partition.dev_id
           << " - Received: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Download TP: " << download_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Record RECEIVE end time (virtual time)
  uint64_t vt_receive_end = VirtualClockNow();

  // Log virtual time event for RECEIVE
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "RECEIVE", "END", vt_receive_start,
                                     vt_receive_end);

  // Log throughput to file
  // DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
  // "DOWNLOAD",
  //                                    size, download_tp, start_us, end_us);

  auto [o_ptr, o_size] = partition.mat[0];
  int64_t row_size = o_size / partition.h_dim / sizeof(float);
  int64_t col_size = partition.h_dim;

  uint64_t ul_overhead = CurrentTimeMicros() - partition.timestamp;

  LOG_DEBUG << part_key << " partition: " << partition.DebugString();

  start = std::chrono::high_resolution_clock::now();
  auto output = torch::from_blob(o_ptr, {row_size, col_size},
                                 FLOAT32_TENSOR_OPTIONS(torch::kCPU));
  {
    // std::lock_guard<std::mutex> lock(outputs_mutex_[partition.oid]);
    auto& output_matrix = reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
                              ->GetOutputMatrix(partition.oid);
    IndexPutMatrixBlock(output_matrix, output, partition.row, partition.col,
                        partition.pivot, ctx_.block_size);
  }
  end = std::chrono::high_resolution_clock::now();
  LOG_DEBUG << "UpdateMatrixBlock time: "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     start)
                   .count()
            << "us";

  // Record RECEIVE end time (virtual time) for RECEIVE phase
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "RECEIVE", "END", vt_receive_start,
                                     vt_receive_end);

  // Mark partition as FINISHED and remove from tracker
  PARTITION_TRACKER.MarkPartitionFinished(part_key);
  PARTITION_TRACKER.RemovePartitionByKey(part_key);
  DEVICE_TRACKER.RecordPartitionProcessed(partition.dev_id);

  reinterpret_cast<ProxySvrImpl*>(ctx_.instance)
      ->IncRspCbCount(partition.oid, 1);
}

// Helper for zero-copy send cleanup: releases ScatterGatherBuffer and
// MatrixPartition when libevent is done sending a segment.
struct ZeroCopySendContext {
  ScatterGatherBufferPtr sg_buffer;
  MatrixPartitionPtr partition;
};

static void ZeroCopySendCleanup(const void* /*data*/, size_t /*len*/,
                                void* arg) {
  // shared_ptr ref count decrements; when last segment cleanup fires, the
  // ZeroCopySendContext (and thus sg_buffer + partition) are freed.
  delete static_cast<std::shared_ptr<ZeroCopySendContext>*>(arg);
}

void ProxySvrHandle::SendInLoop(const ConnectionUeventPtr& conn,
                                const MatrixPartitionPtr partition) {
  // check connection valid first
  if (conn->IsClosed()) {
    LOG_ERROR << "Connection to " << conn->GetPeerAddress().ToString()
              << " is not valid. Cannot send partition.";
    return;
  }

  string client_addr = conn->GetPeerAddress().ToString();

  task_queue_.push_back([this, conn, partition, client_addr]() {
    // Record SEND start time (virtual time)
    uint64_t vt_send_start = VirtualClockNow();
    DEVICE_TRACKER.LogVirtualTimeEvent(partition->dev_id, partition->gemm_id,
                                       "SEND", "START", vt_send_start,
                                       vt_send_start);

    // Zero-copy scatter-gather serialization (avoids tensor memcpy)
    auto t_serialize_start = std::chrono::high_resolution_clock::now();
    auto sg_buffer = partition->SerializeZeroCopy();
    auto t_serialize_end = std::chrono::high_resolution_clock::now();
    auto serialize_us = std::chrono::duration_cast<std::chrono::microseconds>(
                            t_serialize_end - t_serialize_start)
                            .count();

    size_t size = sg_buffer->GetTotalSize();

    // Create shared context to keep scatter-gather buffer and partition alive
    // until all segments are sent
    auto ctx = std::make_shared<ZeroCopySendContext>();
    ctx->sg_buffer = sg_buffer;
    ctx->partition = partition;

    // Zero-copy send each segment
    auto t_send_start = std::chrono::high_resolution_clock::now();
    for (const auto& segment : sg_buffer->GetSegments()) {
      // Each cleanup callback holds a shared_ptr copy of the context
      auto* ref = new std::shared_ptr<ZeroCopySendContext>(ctx);
      conn->SendDataZeroCopy(segment.data, segment.size, ZeroCopySendCleanup,
                             ref);
    }
    auto t_send_end = std::chrono::high_resolution_clock::now();
    auto send_us = std::chrono::duration_cast<std::chrono::microseconds>(
                       t_send_end - t_send_start)
                       .count();

    conn_inflight_[client_addr] += 1;

    double actual_send_tp_bs =
        (send_us > 0) ? (size * 1000000.0 / send_us) : 0.0;
    double actual_send_tp_gbs = actual_send_tp_bs / (1024.0 * 1024.0 * 1024.0);
    LOG_INFO << "[SendInLoop-Timing] Device " << partition->dev_id
             << ", gemm_id=" << partition->gemm_id << ", Size: " << size
             << " bytes"
             << " | Serialize(ZC): " << serialize_us << " us"
             << ", SendData(ZC): " << send_us << " us"
             << ", Actual TP: " << actual_send_tp_gbs << " GB/s";

    // Record SEND end time (virtual time)
    uint64_t vt_send_end = VirtualClockNow();
    DEVICE_TRACKER.LogVirtualTimeEvent(partition->dev_id, partition->gemm_id,
                                       "SEND", "END", vt_send_start,
                                       vt_send_end);

    DEVICE_TRACKER.RecordBytesSent(partition->dev_id, size);

    double last_packet_tp =
        DEVICE_TRACKER.GetLastPacketThroughput(partition->dev_id);
    uint64_t start_us, end_us;
    DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition->dev_id, start_us,
                                                end_us);
    DEVICE_TRACKER.LogThroughputToFile(partition->dev_id, partition->gemm_id,
                                       "SEND", size, last_packet_tp, start_us,
                                       end_us);
  });

  if (conn_inflight_[client_addr] >= ctx_.max_inflight) {
    return;
  }

  auto task = task_queue_.front();
  task_queue_.pop_front();
  task();
}

void ProxySvrHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected from " << client_addr;
  conn_inflight_[client_addr] = 0;

  // Send registration request to client
  SendRegisterRequest(conn);
}

void ProxySvrHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected from " << client_addr;
  conn_inflight_.erase(client_addr);
  // device_info_.erase(client_addr);

  std::string conn_addr = conn->GetPeerAddress().ToString();

  // Find device ID by connection address
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(conn_addr);

  LOG_INFO << "[ConnectionClosedCb] Device " << device_id
           << " (addr: " << conn_addr << ") disconnected";

  // Step 1: Check if failed device has pending partitions
  bool has_pending_partitions =
      PARTITION_TRACKER.HasPendingPartitions(device_id);
  size_t pending_count = PARTITION_TRACKER.GetDevicePartitionCount(device_id);

  if (has_pending_partitions) {
    LOG_WARN << "[ConnectionClosedCb] Device " << device_id << " failed with "
             << pending_count << " pending partitions";
  }

  // Step 2: Mark all running partitions as failed
  PARTITION_TRACKER.MarkDevicePartitionsFailed(device_id);

  // Step 3: Remove connection from maps
  // conn_map_.erase(conn_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.RemoveDeviceConnection(device_id);
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }
}

/********************************ProxySvrImpl****************************************/

ProxySvrImpl::ProxySvrImpl(ProxyEnvCfg& ctx)
    : ctx_(ctx), listener_(nullptr), rsp_cb_counts_(5) {
  // Initialize with greedy scheduling policy by default
}

ProxySvrImpl::~ProxySvrImpl() {
  LOG_INFO << "[ProxySvrImpl::~ProxySvrImpl] Shutting down ProxySvrImpl";
  loop_->CancelTimer(failed_partition_check_timer_);
}

void ProxySvrImpl::Initialize(UeventLoop* loop) {
  LOG_INFO << "[ProxySvrImpl::Initialize] Starting server initialization";
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - listen_ip=" << ctx_.listen_ip
           << ", listen_port=" << ctx_.listen_port;
  LOG_INFO << "[ProxySvrImpl::Initialize] Config - num_device="
           << ctx_.num_device << ", thread=" << ctx_.thread;

  // Initialize virtual clock
  base::VirtualClock::instance().Initialize();
  LOG_INFO << "[ProxySvrImpl::Initialize] Virtual clock initialized";

  // Initialize performance logging (server side)
  DEVICE_TRACKER.InitSeparatePerfLog("./logs", "server");
  LOG_INFO << "[ProxySvrImpl::Initialize] Performance logging initialized at "
              "./logs/perf_server.log";

  loop_ = loop;

  auto create_handle_cb = bind(ProxySvrHandle::CreateMyself, ref(ctx_), _1);
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  listener_ =
      make_shared<ListenerLibevent>(loop, addr, "proxy_listener", Option());
  listener_->SetCreateLoopHandleCb(create_handle_cb);
  listener_->SetThreadInitCb(ProxySvrHandle::ThreadInit);
  listener_->SetConnectionSuccessCb(
      bind(&ProxySvrImpl::ConnectionSuccessCb, shared_from_this(), _1));
  listener_->SetMessageReadCb(
      bind(&ProxySvrImpl::RequestCb, shared_from_this(), _1));
  listener_->SetConnectionClosedCb(
      bind(&ProxySvrImpl::ConnectionClosedCb, shared_from_this(), _1));
  listener_->SetMessageWriteCb([](const ConnectionUeventPtr& conn) {});
  listener_->SetThreadNum(ctx_.thread);
  listener_->StartPrimaryLoop();

  LOG_INFO << "[ProxySvrImpl::Initialize] ProxySvrImpl listen on:"
           << ctx_.listen_ip << ":" << ctx_.listen_port;

  ctx_.instance = this;

  // Start();
  // InitLogger();

  // no more than 5 MAtMul in parallel
  outputs_ = std::move(std::vector<torch::Tensor>(5));
  // rsp_cb_counts_ = std::move(std::vector<std::atomic_ullong>(5));
  // outputs_mutex_ = std::move(std::vector<std::mutex>(5));
  for (int i = 0; i < 5; i++) {
    outputs_[i] = torch::empty({0, 0});
    rsp_cb_counts_[i] = 0;
  }

  LOG_INFO << "[ProxySvrImpl::Initialize] Server initialization completed. "
              "Waiting for connections...";

  // Start periodic partition health check (every 0.1 seconds)
  // failed_partition_check_timer_ = loop->RunEvery(
  //     0.1, std::bind(&ProxySvrImpl::CheckFailedPartitions, this));
  // idle_partition_redistribute_timer_ =
  //     loop->RunEvery(0.5, std::bind(&ProxySvrImpl::SendIdlePartitions,
  //     this));
  LOG_INFO << "[ProxySvrImpl::Initialize] Started periodic partition health "
              "check (interval=0.1s)";
}

void ProxySvrImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
  // loop->RunInLoop(bind(&ProxySvrHandle::ConnectionSuccessCb, handle, conn));
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));
}

void ProxySvrImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));

  // Unregister device from tracker
  string client_addr = conn->GetPeerAddress().ToString();
  int64_t device_id = DEVICE_TRACKER.GetDeviceIdByAddr(client_addr);
  if (device_id != -1) {
    DEVICE_TRACKER.UnregisterDevice(device_id);
  }

  // loop->RunInLoop(bind(&ProxySvrHandle::ConnectionClosedCb, handle, conn));

  // LOG_INFO << "[ConnectionClosedCb] Connection removed. Remaining
  // connections: "
  //          << conn_map_.size();
  // LOG_INFO << "[ConnectionClosedCb] Connection map contents:";
  // for (const auto& conn_pair : conn_map_) {
  //   LOG_INFO << "  - " << conn_pair.first << " -> "
  //            << (conn_pair.second ? "valid" : "null");
  // }

  // // Step 4: Handle partition redistribution if needed
  // if (has_pending_partitions && conn_map_.size() > 0) {
  //   LOG_INFO << "[ConnectionClosedCb] Starting partition redistribution for "
  //               "failed device "
  //            << device_id;
  //   HandleDeviceFailure(device_id);
  //   LOG_INFO << "[ConnectionClosedCb] Partition redistribution completed";
  // } else if (has_pending_partitions && conn_map_.empty()) {
  //   LOG_ERROR << "[ConnectionClosedCb] Device " << device_id << " failed with
  //   "
  //             << pending_count
  //             << " pending partitions but no other devices available!";
  // }
}

void ProxySvrImpl::RequestWriteCb(const uevent::ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  handle->RequestWriteCb(conn);
}

void ProxySvrImpl::RequestCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
  loop->RunInLoop(bind(&ProxySvrHandle::RequestCb, handle, conn));
}

// void ProxySvrImpl::RequestCb(const uevent::ConnectionUeventPtr& conn) {
//   size_t readable = conn->ReadableLength();

//   int ret = 0;
//   size_t packsize;
//   ret = conn->ReceiveData(&packsize, sizeof(size_t));
//   if (ret < 0) {
//     conn->ForceClose();
//     LOG_ERROR << "ReceiveData on a closed connection";
//     return;
//   }

//   auto* loop = conn->GetLoop();
//   loop->AssertInLoopThread();
//   auto* loop_handle = loop->GetLoopHandle();
//   auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_handle);
//   handle->RequestCb(conn);
// }

void ProxySvrImpl::DispatchMatMulAsync(torch::Tensor& mat_a,
                                       torch::Tensor& mat_b) {
  LOG_INFO << "[DispatchMatMulAsync] Starting dispatch - mm_count="
           << mm_count_;

  outputs_[mm_count_].set_data(CreateOutputMatrix(mat_a, mat_b));
  auto partitions = PartitionMatrices(mat_a, mat_b, ctx_.block_size);
  auto a_shape = mat_a.sizes().vec();
  auto b_shape = mat_b.sizes().vec();

  if (partitions.empty()) {
    LOG_ERROR << "[DispatchMatMulAsync] No partitions generated!";
    return;
  }

  auto cur_ver = partitions[0]->version;
  LOG_INFO << "[" << cur_ver << "] Number of partitions: " << partitions.size()
           << " for A: " << a_shape << " and B: " << b_shape;

  LOG_INFO << "[DispatchMatMulAsync] Before random_shuffle - partitions.size()="
           << partitions.size();
  std::random_shuffle(partitions.begin(), partitions.end());

  // LOG_INFO
  //     << "[DispatchMatMulAsync] Before RephrasePartitions - ctx_.num_device="
  //     << ctx_.num_device;

  // RephrasePartitions(partitions);

  // LOG_INFO << "[DispatchMatMulAsync] After RephrasePartitions - created "
  //          << partitions.size() << " partitions";
  auto start = std::chrono::high_resolution_clock::now();

  DecRspCbCount(mm_count_, partitions.size());

  LOG_INFO << "[DispatchMatMulAsync] Creating " << partitions.size()
           << " partitions as IDLE";

  // Add all partitions to tracker as IDLE - they will be dispatched by
  // SendIdlePartitions
  for (auto& partition : partitions) {
    partition->oid = mm_count_;
    partition->gemm_id = gemm_id_count_;  // assign global gemm_id
    partition->dev_id =
        -1;  // Mark as unassigned, will be assigned by scheduling policy

    // Add partition to tracker with dev_id=-1 (unassigned, to be scheduled)
    // The tracker will use owner_device_id=-1 until scheduling assigns a real
    // device
    PARTITION_TRACKER.AddPartition(-1, partition->GetPartitionKey(), mm_count_,
                                   partition);

    LOG_DEBUG << "[DispatchMatMulAsync] Created IDLE partition key="
              << partition->GetPartitionKey()
              << ", dev_id=" << partition->dev_id << " (unassigned)"
              << ", oid=" << mm_count_ << ", gemm_id=" << partition->gemm_id;
  }
  auto end = std::chrono::high_resolution_clock::now();
  LOG_INFO << "[DispatchMatMulAsync] Created " << partitions.size()
           << " IDLE partitions in "
           << std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count()
           << "us. Partitions will be sent by SendIdlePartitions timer. "
              "gemm_id_count="
           << gemm_id_count_;
  mm_count_++;
  gemm_id_count_++;  // increment global gemm_id for next operation

  auto* handle = reinterpret_cast<ProxySvrHandle*>(loop_->GetLoopHandle());
  loop_->QueueInLoop(bind(&ProxySvrHandle::SendIdlePartitions, handle));
}

torch::Tensor ProxySvrImpl::WaitMatMul(int oid) {
  auto start = std::chrono::high_resolution_clock::now();
  LOG_INFO << "[WaitMatMul] Starting wait for oid=" << oid
           << ", rsp_cb_counts_[oid]=" << rsp_cb_counts_[oid];

  int poll_count = 0;
  while (rsp_cb_counts_[oid] > 0) {
    poll_count++;
    if (poll_count % 50 == 0) {  // Log every 5 seconds (50 * 100ms)
      LOG_WARN << "[WaitMatMul] Still waiting for oid=" << oid
               << ", rsp_cb_counts_[oid]=" << rsp_cb_counts_[oid]
               << ", poll_count=" << poll_count * 100 << "ms";

      // Diagnose partition states across all devices
      auto connected_devices = DEVICE_TRACKER.GetConnectedDevices();
      LOG_WARN << "[WaitMatMul] Connected devices: "
               << connected_devices.size();

      size_t total_idle = 0, total_running = 0, total_finished = 0;
      size_t devices_with_partitions = 0;
      std::vector<int64_t> devices_with_oid;

      for (int64_t device_id : connected_devices) {
        auto stats = PARTITION_TRACKER.GetDeviceOidStats(device_id, oid);
        size_t device_total =
            stats.idle_count + stats.running_count + stats.finished_count;

        if (device_total > 0) {
          devices_with_partitions++;
          devices_with_oid.push_back(device_id);
          bool is_connected = DEVICE_TRACKER.IsDeviceConnected(device_id);
          total_idle += stats.idle_count;
          total_running += stats.running_count;
          total_finished += stats.finished_count;

          LOG_WARN << "[WaitMatMul]   Device " << device_id
                   << " (connected=" << (is_connected ? "YES" : "NO") << ")"
                   << ": IDLE=" << stats.idle_count
                   << ", RUNNING=" << stats.running_count
                   << ", FINISHED=" << stats.finished_count
                   << ", Total=" << device_total;

          // Show first few partition keys for debugging (only for first 3
          // devices)
          if (devices_with_partitions <= 3) {
            if (!stats.partition_keys.empty() &&
                stats.partition_keys.size() <= 5) {
              std::string keys_str;
              for (const auto& key : stats.partition_keys) {
                if (!keys_str.empty()) keys_str += ", ";
                keys_str += key;
              }
              LOG_WARN << "[WaitMatMul]     Partition keys: " << keys_str;
            } else if (stats.partition_keys.size() > 5) {
              LOG_WARN << "[WaitMatMul]     First partition key: "
                       << stats.partition_keys[0] << " (+ "
                       << (stats.partition_keys.size() - 1) << " more)";
            }
          }
        }
      }

      // Summary with device distribution info
      LOG_WARN << "[WaitMatMul] Summary for oid=" << oid
               << ": Devices with partitions=" << devices_with_partitions << "/"
               << connected_devices.size() << ", Total IDLE=" << total_idle
               << ", RUNNING=" << total_running
               << ", FINISHED=" << total_finished
               << ", Expected remaining=" << rsp_cb_counts_[oid];

      // Critical: if only 1 device has all partitions, this is a scheduling
      // problem!
      if (devices_with_partitions == 1 && connected_devices.size() > 1) {
        LOG_ERROR << "[WaitMatMul] ⚠️  SCHEDULING ISSUE: All "
                  << (total_idle + total_running + total_finished)
                  << " partitions assigned to single device "
                  << devices_with_oid[0] << " while "
                  << (connected_devices.size() - 1)
                  << " other devices are idle!";
      } else if (devices_with_partitions > 0 && devices_with_partitions <= 10) {
        std::string device_list;
        for (auto dev_id : devices_with_oid) {
          if (!device_list.empty()) device_list += ", ";
          device_list += std::to_string(dev_id);
        }
        LOG_WARN << "[WaitMatMul] Devices with oid=" << oid << ": ["
                 << device_list << "]";
      }

      // Check if partitions are stuck in RUNNING state
      if (total_running > 0 && total_running == rsp_cb_counts_[oid] &&
          total_idle == 0) {
        LOG_ERROR << "[WaitMatMul] ⚠️  STUCK PARTITIONS: All " << total_running
                  << " partitions stuck in RUNNING state for "
                  << poll_count * 100 << "ms";
        LOG_ERROR << "[WaitMatMul] Possible causes: 1) Devices not responding "
                     "2) Network issues 3) Devices processing too slowly";

        // Sample a few devices to check connection quality
        size_t check_count = std::min(size_t(5), devices_with_oid.size());
        for (size_t i = 0; i < check_count; ++i) {
          int64_t dev_id = devices_with_oid[i];
          auto conn = DEVICE_TRACKER.GetDeviceConnection(dev_id);
          bool has_conn = (conn != nullptr);
          bool conn_closed = has_conn ? conn->IsClosed() : true;
          LOG_ERROR << "[WaitMatMul]   Sample device " << dev_id
                    << ": has_connection=" << (has_conn ? "YES" : "NO")
                    << ", connection_closed=" << (conn_closed ? "YES" : "NO");
        }
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto shape = outputs_[oid].sizes().vec();
  auto wait_time =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start)
          .count();
  LOG_INFO << "[WaitMatMul] Completed for oid=" << oid
           << ", wait_time=" << wait_time << "us, shape=" << shape;

  mm_count_--;
  return outputs_[oid];
}

void ProxySvrImpl::IncRspCbCount(int oid, size_t count) {
  int prev = rsp_cb_counts_[oid];
  rsp_cb_counts_[oid] -= count;
  LOG_DEBUG << "[IncRspCbCount] oid=" << oid << ", count=" << count
            << ", prev=" << prev << ", now=" << rsp_cb_counts_[oid];
}

void ProxySvrImpl::RephrasePartitions(
    std::vector<MatrixPartitionPtr>& partitions,
    const std::unordered_set<int64_t>& excluded_devices) {
  // Get connected devices from tracker
  auto& tracker = DEVICE_TRACKER;
  std::vector<int64_t> device_ids = tracker.GetConnectedDevices();

  int actual_num_devices = static_cast<int>(device_ids.size());
  LOG_INFO << "[RephrasePartitions] Starting with " << partitions.size()
           << " partitions, actual connected devices=" << actual_num_devices
           << ", excluded_devices=" << excluded_devices.size();

  if (actual_num_devices == 0) {
    LOG_ERROR << "[RephrasePartitions] No devices connected!";
    return;
  }

  std::sort(device_ids.begin(), device_ids.end());

  auto assignments = ctx_.sched_policy->AssignPartitionsToDevices(
      partitions, excluded_devices);

  if (assignments.size() != partitions.size()) {
    LOG_ERROR << "[RephrasePartitions] Policy returned " << assignments.size()
              << " assignments for " << partitions.size() << " partitions";
    return;
  }

  // Apply assignments to partitions
  for (size_t i = 0; i < partitions.size(); ++i) {
    partitions[i]->dev_id = assignments[i];
    LOG_DEBUG << "[RephrasePartitions] Partition " << i
              << " assigned to device_id " << assignments[i];
  }

  LOG_INFO << "[RephrasePartitions] Completed partitioning";
}

void ProxySvrImpl::HandleDeviceFailure(int64_t failed_device_id) {
  // Get partitions before redistribution to count OIDs
  auto failed_partitions =
      PARTITION_TRACKER.GetDevicePartitions(failed_device_id);

  if (failed_partitions.empty()) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no pending partitions to redistribute";
    return;
  }

  // Count FAILED partitions and OIDs (only those that were RUNNING)
  size_t num_failed_partitions = 0;
  std::unordered_map<int64_t, size_t> oid_counts;
  for (const auto& part : failed_partitions) {
    if (part->state == PartitionState::IDLE) {
      num_failed_partitions++;
      oid_counts[part->oid]++;
    }
  }

  if (num_failed_partitions == 0) {
    LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
             << " has no FAILED partitions to redistribute";
    return;
  }

  LOG_INFO << "[HandleDeviceFailure] Device " << failed_device_id
           << " failed with " << num_failed_partitions
           << " partitions. Redistributing across all connected devices";

  LOG_INFO << "[HandleDeviceFailure] OID breakdown for failed partitions:";
  for (const auto& [oid, count] : oid_counts) {
    LOG_INFO << "  - OID " << oid << ": " << count << " partitions";
  }

  // Redistribute partitions across all connected devices
  // PARTITION_TRACKER.RedistributeFailedDevicePartitions(failed_device_id);

  // Get connected devices to send redistributed partitions to them
  std::vector<int64_t> connected_devices = DEVICE_TRACKER.GetConnectedDevices();

  // CRITICAL: Decrement response counters for partitions from failed device
  // These partitions were in-flight when the device failed, so they will never
  // produce responses. We must decrement their response counters to prevent
  // WaitMatMul from hanging forever.
  for (const auto& [oid, count] : oid_counts) {
    LOG_INFO << "[HandleDeviceFailure] Decrementing response counter for OID "
             << oid << " by " << count << " (in-flight partitions lost)";
    for (size_t i = 0; i < count; ++i) {
      IncRspCbCount(oid, 1);  // Decrement the counter
    }
  }

  LOG_INFO << "[HandleDeviceFailure] Completed failure handling for device "
           << failed_device_id;
}

void ProxySvrHandle::SendIdlePartitions() {
  auto idle_partitions = PARTITION_TRACKER.GetIdlePartitions();

  if (idle_partitions.empty()) {
    LOG_DEBUG << "[SendIdlePartitions] No IDLE partitions to send";
    return;
  }

  LOG_INFO << "[SendIdlePartitions] Found " << idle_partitions.size()
           << " IDLE partitions, running scheduling policy";

  auto redistributed =
      ctx_.sched_policy->RedistributePartitions(idle_partitions);

  if (redistributed.empty()) {
    LOG_DEBUG << "[SendIdlePartitions] No available devices for redistribution";
    return;
  }

  LOG_INFO << "[SendIdlePartitions] Scheduling complete, moving partitions to "
              "assigned devices";

  // IMPORTANT: After scheduling, partitions' owner_device_id has been updated
  // We need to move them from device -1 to their assigned devices in tracker
  // Do this BEFORE marking as RUNNING to avoid race conditions
  for (const auto& part_info : idle_partitions) {
    int64_t old_device =
        part_info
            ->owner_device_id;  // This might be wrong due to scheduling update
    // Find old device from partition_to_device_ map

    LOG_DEBUG << "[SendIdlePartitions] Moving partition " << part_info->key
              << " (oid=" << part_info->oid << ") to device "
              << part_info->owner_device_id;

    // Remove from old location and add to new location
    PARTITION_TRACKER.RemovePartitionByKey(part_info->key);
    PARTITION_TRACKER.AddPartition(part_info->owner_device_id, part_info->key,
                                   part_info->oid, part_info->partition);

    // Update partition's dev_id to match the assigned device
    part_info->partition->dev_id = part_info->owner_device_id;

    // Mark partition as RUNNING after it's in the correct device list
    PARTITION_TRACKER.MarkPartitionRunning(part_info->key);
  }

  LOG_INFO << "[SendIdlePartitions] Sending " << idle_partitions.size()
           << " partitions to devices";

  // Send each partition
  for (const auto& part_info : idle_partitions) {
    LOG_DEBUG << "[SendIdlePartitions] Sending partition " << part_info->key
              << " to device " << part_info->owner_device_id;

    auto target_conn =
        DEVICE_TRACKER.GetDeviceConnection(part_info->owner_device_id);
    if (!target_conn) {
      LOG_ERROR << "[SendIdlePartitions] No connection for device "
                << part_info->owner_device_id;
      // Mark as IDLE again so it can be rescheduled
      PARTITION_TRACKER.MarkPartitionIdle(part_info->key);
      continue;
    }
    auto* loop = target_conn->GetLoop();
    auto* handle = reinterpret_cast<ProxySvrHandle*>(loop->GetLoopHandle());
    loop->RunInLoop(bind(&ProxySvrHandle::SendInLoop, handle, target_conn,
                         part_info->partition));
  }

  LOG_INFO << "[SendIdlePartitions] Completed sending "
           << idle_partitions.size() << " partitions to devices";
}

void ProxySvrImpl::CheckFailedPartitions() {
  auto& tracker = DEVICE_TRACKER;

  // Get all devices
  std::vector<int64_t> all_devices = tracker.GetAllDevices();

  LOG_DEBUG << "[CheckFailedPartitions] Checking " << all_devices.size()
            << " devices for failed partitions";

  for (int64_t device_id : all_devices) {
    // Skip connected devices
    if (tracker.IsDeviceConnected(device_id)) {
      continue;
    }

    // Check if disconnected device has pending partitions
    if (PARTITION_TRACKER.HasPendingPartitions(device_id)) {
      size_t pending_count =
          PARTITION_TRACKER.GetDevicePartitionCount(device_id);
      LOG_WARN << "[CheckFailedPartitions] Detected disconnected device "
               << device_id << " with " << pending_count
               << " pending partitions. Initiating failure handling.";

      HandleDeviceFailure(device_id);
    }
  }
}

/*********************************ProxySvr***************************************/
typedef ProxySvr::Status ProxyStatus;
typedef ProxyStatus::StatusType ProxyStatusType;
const map<ProxyStatusType, string> ProxyStatus::status_str_ = {
    {kOK, ""},
    {kFatal, "Proxy svr fatal:"},
    {kUnknown, "Proxy svr unknown:"},
};

ProxySvr::ProxySvr() : svr_(nullptr), loop_thread_(nullptr) {}

void ProxySvr::Initialize(const std::string& cfg_file) {
  context_.Initialize(cfg_file);
  svr_ = make_shared<ProxySvrImpl>(context_);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxySvrHandle::CreateMyself, ref(context_), _1),
      bind(&ProxySvrImpl::Initialize, svr_, _1), "Proxy svr main thread");
}

void ProxySvr::Start() { loop_thread_->StartLoop(); }

}  // namespace backend
}  // namespace morphling
