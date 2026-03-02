#include "proxy_cli.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "base/my_uuid.h"
#include "common/generator.h"
#include "common/stats.h"
#include "device_tracker.h"
#include "network/eventloop_libevent.h"
#include "network/ueventloop_thread_pool.h"
#include "proto_base.h"
#include "scheduler/cpu_worker.h"
#include "scheduler/gpu_worker.h"
#include "utils/cuda_utils.h"
#include "utils/logger.h"

using namespace std;
using namespace std::placeholders;
using namespace uevent;

#include <sys/stat.h>

#include <iostream>
#include <set>

namespace morphling {
namespace backend {

int CudaPinBuffer(void* ptr, size_t size) {
  auto err = cudaHostRegister(ptr, size, cudaHostRegisterMapped);
  return (err == cudaSuccess) ? 0 : -1;
}

void CudaUnpinBuffer(void* ptr, size_t size) { cudaHostUnregister(ptr); }

int PosixPinBuffer(void* ptr, size_t size) {
  // No-op for CPU memory, but could add mlock here if desired
  return mlock(ptr, size);
}

void PosixUnpinBuffer(void* ptr, size_t size) { munlock(ptr, size); }

/*********************************ProxyCliHandle************************************/

ProxyCliHandle::ProxyCliHandle(ProxyEnvCfg& ctx, UeventLoop* loop,
                               int64_t device_id,
                               XtGemmWorkerPool* gpu_pool,
                               CpuWorkerPool* cpu_pool)
    : ctx_(ctx),
      loop_(loop),
      device_id_(device_id),
      gpu_pool_(gpu_pool),
      cpu_pool_(cpu_pool) {
  SRV_STATS->Initialize();
}

ProxyCliHandle::~ProxyCliHandle() = default;

void ProxyCliHandle::ThreadInit(uevent::UeventLoop* loop) {
  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  // handle->RegisterService();
}

// void ProxyCliHandle::RegisterService() {
//   proxy_cbs_.insert(make_pair(NFSPROC4_NULL, NullService::CreateMyself));
//   proxy_cbs_.insert(
//       make_pair(NFSPROC4_COMPOUND, CompoundService::CreateMyself));
// }

// Context for zero-copy scatter-gather send from client.
// Ensures referenced memory (scatter-gather buffer + CUDA result buffer)
// stays alive until libevent finishes sending all segments.
struct ResponseSendContext {
  ScatterGatherBufferPtr sg_buffer;
  // Deferred CUDA managed memory release (result buffer)
  void* cuda_ptr = nullptr;
  // Deferred host memory release (CPU pool result buffer)
  void* host_ptr = nullptr;

  ~ResponseSendContext() {
    if (cuda_ptr) {
      cudaFree(cuda_ptr);
    }
    if (host_ptr) {
      free(host_ptr);
    }
  }
};

static void ResponseSendCleanup(const void* /*data*/, size_t /*len*/,
                                void* arg) {
  delete static_cast<std::shared_ptr<ResponseSendContext>*>(arg);
}

// Helper for zero-copy send cleanup of SerializationBuffer (registration msgs)
static void SerializationBufferSendCleanup(const void* /*data*/,
                                           size_t /*len*/, void* arg) {
  delete static_cast<SerializationBufferPtr*>(arg);
}

void ProxyCliHandle::ResponseToCaller(const uevent::ConnectionUeventPtr& conn,
                                      const MatrixPartition& partition,
                                      void* deferred_cuda_ptr,
                                      void* deferred_host_ptr) {
  assert(conn);

  string client_addr = conn->GetPeerAddress().ToString();
  if (conn->IsClosed()) {
    LOG_WARN << DEV_TAG(device_id_, partition.gemm_id)
             << "connection already closed:" << client_addr;
    // Release deferred memory since we won't send
    if (deferred_cuda_ptr) {
      cudaFree(deferred_cuda_ptr);
    }
    if (deferred_host_ptr) {
      free(deferred_host_ptr);
    }
    return;
  }

  // Record UPLOAD start time (virtual time)
  uint64_t vt_upload_start = VirtualClockNow();
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", "START", vt_upload_start,
                                     vt_upload_start);

  // Zero-copy scatter-gather serialization (avoids tensor memcpy)
  auto sg_buffer = partition.SerializeZeroCopy();
  auto size = sg_buffer->GetTotalSize();

  // Create send context to keep scatter-gather buffer and cuda managed memory
  // alive until libevent finishes sending all segments
  auto ctx = std::make_shared<ResponseSendContext>();
  ctx->sg_buffer = sg_buffer;
  ctx->cuda_ptr = deferred_cuda_ptr;
  ctx->host_ptr = deferred_host_ptr;

  // Zero-copy send each segment
  for (const auto& segment : sg_buffer->GetSegments()) {
    auto* ref = new std::shared_ptr<ResponseSendContext>(ctx);
    conn->SendDataZeroCopy(segment.data, segment.size, ResponseSendCleanup,
                           ref);
  }

  // Record UPLOAD end time (virtual time)
  uint64_t vt_upload_end = VirtualClockNow();

  // Record bytes sent (upload response back to server)
  DEVICE_TRACKER.RecordBytesSent(partition.dev_id, size);

  // Log upload throughput after sending response
  double upload_tp = DEVICE_TRACKER.GetUploadThroughput(partition.dev_id);
  double last_packet_tp =
      DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp =
      DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();

  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us,
                                              end_us);

  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[ResponseToCaller] Sent: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Upload TP: " << upload_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Log virtual time event for UPLOAD
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", "END", vt_upload_start,
                                     vt_upload_end);

  // Log throughput to file
  DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
                                     "UPLOAD", size, upload_tp, start_us,
                                     end_us);

  RECORD_SRV_COUNT(SRV_TOTAL_QUERY, 1);
  RECORD_SRV_COUNT(SRV_TOTAL_TRAFFIC, size);

  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << "Response sent to "
            << client_addr << ", size: " << size;
}

bool ProxyCliHandle::ShouldUseGpu() const {
  if (!gpu_pool_) return false;
  if (!cpu_pool_) return true;
  // Both pools: compare estimated wait = depth * avg_duration
  int64_t gpu_wait = static_cast<int64_t>(gpu_pool_->GetPendingTaskCount())
                   * gpu_duration_tracker_.GetAverageDurationUs();
  int64_t cpu_wait = static_cast<int64_t>(cpu_pool_->GetPendingTaskCount())
                   * cpu_duration_tracker_.GetAverageDurationUs();
  LOG_DEBUG << "[PoolDispatch] gpu_depth="
            << gpu_pool_->GetPendingTaskCount()
            << " gpu_avg=" << gpu_duration_tracker_.GetAverageDurationUs()
            << "us gpu_wait=" << gpu_wait
            << "us | cpu_depth="
            << cpu_pool_->GetPendingTaskCount()
            << " cpu_avg=" << cpu_duration_tracker_.GetAverageDurationUs()
            << "us cpu_wait=" << cpu_wait << "us";
  return gpu_wait <= cpu_wait;  // prefer GPU on tie
}

void ProxyCliHandle::HandlePartition(const uevent::ConnectionUeventPtr& conn,
                                     const MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();

  // Record COMPUTE start time (virtual time)
  uint64_t vt_compute_start = VirtualClockNow();
  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[HandlePartition] COMPUTE START"
           << ", vt_start=" << vt_compute_start;
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "COMPUTE", "START", vt_compute_start,
                                     vt_compute_start);

  // create tensors from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  auto* row_ptr = reinterpret_cast<const float*>(r_ptr);
  auto* col_ptr = reinterpret_cast<const float*>(c_ptr);

  LOG_DEBUG << part_key << " Row: [" << row_size << ", " << partition.h_dim
            << "], Col: [" << col_size << ", " << partition.h_dim << "]";

  size_t out_elems =
      static_cast<size_t>(row_size) * static_cast<size_t>(col_size);
  int64_t out_bytes = static_cast<int64_t>(out_elems * sizeof(float));

  auto task_enqueue_time = SlidingWindowDurationTracker<>::Now();

  if (ShouldUseGpu()) {
    float* out_ptr = nullptr;
    if (!LogCudaError(cudaMallocManaged(&out_ptr, out_bytes),
                      "cudaMallocManaged(output)")) {
      return;
    }
    SubmitToGpuPool(conn, partition, out_ptr, out_bytes,
                    row_ptr, row_size, col_ptr, col_size,
                    partition.h_dim, vt_compute_start,
                    /*is_host_alloc=*/false, task_enqueue_time);
    return;
  }

  if (cpu_pool_) {
    float* out_ptr = static_cast<float*>(malloc(out_bytes));
    if (!out_ptr) {
      LOG_ERROR << "malloc failed for output buffer (" << out_bytes << " bytes)";
      return;
    }
    SubmitToCpuPool(conn, partition, out_ptr, out_bytes,
                    row_ptr, row_size, col_ptr, col_size,
                    partition.h_dim, vt_compute_start,
                    /*is_host_alloc=*/true, task_enqueue_time);
    return;
  }

  LOG_ERROR << "No worker pool available";
}

std::shared_ptr<GemmArgs> ProxyCliHandle::BuildGemmArgs(
    const float* col_ptr, int64_t col_size,
    const float* row_ptr, int64_t row_size,
    int64_t h_dim, float* out_ptr) {
  auto args = std::make_shared<GemmArgs>();
  args->group_size = 1;
  args->transa[0] = 'T';
  args->transb[0] = 'N';
  args->m[0] = static_cast<int>(col_size);
  args->n[0] = static_cast<int>(row_size);
  args->k[0] = static_cast<int>(h_dim);
  args->alpha[0] = 1.0f;
  args->beta[0] = 0.0f;
  args->a[0] = col_ptr;
  args->lda[0] = static_cast<int>(h_dim);
  args->b[0] = row_ptr;
  args->ldb[0] = static_cast<int>(h_dim);
  args->c[0] = out_ptr;
  args->ldc[0] = static_cast<int>(col_size);
  return args;
}

void ProxyCliHandle::SubmitToGpuPool(
    const uevent::ConnectionUeventPtr& conn,
    const MatrixPartition& partition,
    float* out_ptr, int64_t out_bytes,
    const float* row_ptr, int64_t row_size,
    const float* col_ptr, int64_t col_size,
    int64_t h_dim, uint64_t vt_compute_start,
    bool is_host_alloc,
    SlidingWindowDurationTracker<>::TimePoint task_enqueue_time) {
  auto args = BuildGemmArgs(col_ptr, col_size, row_ptr, row_size,
                            h_dim, out_ptr);

  uint64_t task_num = task_counter_.fetch_add(1, std::memory_order_relaxed);
  std::string task_id = "gemm_" + std::to_string(task_num);

  // Copy partition by value for the callback closure
  MatrixPartition part_copy = partition;

  TaskCallback callback = [this, conn, part_copy, out_ptr, out_bytes,
                           col_size, vt_compute_start,
                           is_host_alloc, task_enqueue_time](const std::string&) {
    loop_->RunInLoop([this, conn, part_copy, out_ptr, out_bytes,
                      col_size, vt_compute_start, is_host_alloc,
                      task_enqueue_time]() {
      OnComputeComplete(conn, part_copy, out_ptr, out_bytes,
                        col_size, vt_compute_start, is_host_alloc,
                        /*is_gpu_task=*/true, task_enqueue_time);
    });
  };

  gpu_pool_->EnqueueGemm(task_id, args, std::move(callback));
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id)
            << "[SubmitToGpuPool] task=" << task_id;
}

void ProxyCliHandle::SubmitToCpuPool(
    const uevent::ConnectionUeventPtr& conn,
    const MatrixPartition& partition,
    float* out_ptr, int64_t out_bytes,
    const float* row_ptr, int64_t row_size,
    const float* col_ptr, int64_t col_size,
    int64_t h_dim, uint64_t vt_compute_start,
    bool is_host_alloc,
    SlidingWindowDurationTracker<>::TimePoint task_enqueue_time) {
  auto args = BuildGemmArgs(col_ptr, col_size, row_ptr, row_size,
                            h_dim, out_ptr);

  uint64_t task_num = task_counter_.fetch_add(1, std::memory_order_relaxed);
  std::string task_id = "gemm_cpu_" + std::to_string(task_num);

  // Copy partition by value for the callback closure
  MatrixPartition part_copy = partition;

  TaskCallback callback = [this, conn, part_copy, out_ptr, out_bytes,
                           col_size, vt_compute_start,
                           is_host_alloc, task_enqueue_time](const std::string&) {
    loop_->RunInLoop([this, conn, part_copy, out_ptr, out_bytes,
                      col_size, vt_compute_start, is_host_alloc,
                      task_enqueue_time]() {
      OnComputeComplete(conn, part_copy, out_ptr, out_bytes,
                        col_size, vt_compute_start, is_host_alloc,
                        /*is_gpu_task=*/false, task_enqueue_time);
    });
  };

  cpu_pool_->EnqueueGemm(task_id, args, std::move(callback));
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id)
            << "[SubmitToCpuPool] task=" << task_id;
}

void ProxyCliHandle::OnComputeComplete(
    const uevent::ConnectionUeventPtr& conn,
    const MatrixPartition& partition,
    float* out_ptr, int64_t out_bytes,
    int64_t col_size, uint64_t vt_compute_start,
    bool is_host_alloc, bool is_gpu_task,
    SlidingWindowDurationTracker<>::TimePoint task_enqueue_time) {
  // Record task turnaround duration for dispatch estimation
  int64_t duration_us =
      SlidingWindowDurationTracker<>::ElapsedUs(task_enqueue_time);
  if (is_gpu_task)
    gpu_duration_tracker_.RecordDuration(duration_us);
  else
    cpu_duration_tracker_.RecordDuration(duration_us);
  LOG_DEBUG << "[PoolDispatch] " << (is_gpu_task ? "GPU" : "CPU")
            << " task done in " << duration_us << "us";

  // Record COMPUTE end time (virtual time)
  uint64_t vt_compute_end = VirtualClockNow();
  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[HandlePartition] COMPUTE END"
           << ", vt_start=" << vt_compute_start
           << ", vt_end=" << vt_compute_end;
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "COMPUTE", "END", vt_compute_start,
                                     vt_compute_end);

  MatrixPartition response = partition;
  response.h_dim = col_size;
  response.timestamp = CurrentTimeMicros();
  response.mat.clear();
  response.mat.push_back({out_ptr, out_bytes});

  // Zero-copy send: result buffer release is deferred to send completion
  if (is_host_alloc) {
    ResponseToCaller(conn, response, /*deferred_cuda_ptr=*/nullptr,
                     /*deferred_host_ptr=*/out_ptr);
  } else {
    ResponseToCaller(conn, response, /*deferred_cuda_ptr=*/out_ptr);
  }
}

void ProxyCliHandle::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "connected to " << client_addr;
}

void ProxyCliHandle::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  string client_addr = conn->GetPeerAddress().ToString();
  LOG_INFO << "disconnected to " << client_addr;
}

/********************************ProxyCliImpl****************************************/

ProxyCliImpl::ProxyCliImpl(ProxyEnvCfg& ctx, int64_t device_id)
    : ctx_(ctx),
      connector_(nullptr),
      cached_tensors_(GB / 16, [](const TensorKey&, const CachedTensor& t) {
        if (t.data) {
          LOG_DEBUG << "Evicting cached tensor, freeing memory: " << t.data;
#ifdef CACHEDTENSOR_CUDA_MALLOC_MANAGED
          cudaFree(t.data);
#else
          free(t.data);
#endif
        }
      }) {
}

void ProxyCliImpl::Initialize(UeventLoop* loop) {
  UsockAddress addr(ctx_.listen_ip, ctx_.listen_port);
  connector_ = make_shared<ConnectorLibevent>(loop, addr, "proxy_connector");
  connector_->SetConnectionSuccessCb(
      bind(&ProxyCliImpl::ConnectionSuccessCb, shared_from_this(), _1));
  connector_->SetConnectionClosedCb(
      bind(&ProxyCliImpl::ConnectionClosedCb, shared_from_this(), _1));
  connector_->SetMessageReadCb(
      bind(&ProxyCliImpl::RequestCb, shared_from_this(), _1));
  connector_->Connect();

  LOG_DEBUG << "ProxyCliImpl connected to:" << ctx_.listen_ip << ":"
            << ctx_.listen_port;

  // Initialize virtual clock
  base::VirtualClock::instance().Initialize();
  LOG_INFO << "[ProxyCliImpl::Initialize] Virtual clock initialized";

  // Initialize performance logging (client side)
  // Client processes requests from all devices, so we use device ID 0 for
  // client-side processing
  DEVICE_TRACKER.InitSeparatePerfLog("./logs", "device", device_id_);
  LOG_INFO << "[ProxyCliImpl::Initialize] Performance logging initialized at "
              "./logs/perf_device_"
           << device_id_ << ".log";

  // Tee logging: write LOG_* output to both console and rotating file
  ::mkdir("./logs", 0755);
  g_tee_log_file = std::make_unique<base::LogFile>("./logs/client_general",
                                                   256 * 1024 * 1024, true, 3);
  base::Logger::setOutput(TeeOutput);
  base::Logger::setFlush(TeeFlush);
  LOG_INFO << "[ProxyCliImpl::Initialize] Tee logging initialized";
}

void ProxyCliImpl::ConnectionSuccessCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->ConnectionSuccessCb(conn);
}

void ProxyCliImpl::ConnectionClosedCb(const ConnectionUeventPtr& conn) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  handle->ConnectionClosedCb(conn);
}

void ProxyCliImpl::RequestCb(const ConnectionUeventPtr& conn) {
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

    // LOG_DEBUG << "packsize: " << packsize << ", datasize: " << datasize
    //           << ", readable: " << readable;
    if (readable < datasize) {
      return;
    }

    // Zero-copy receive: get contiguous pointer into evbuffer
    unsigned char* raw_data = conn->PullupData(datasize);
    if (raw_data == nullptr) {
      LOG_ERROR << "PullupData failed for size " << datasize;
      return;
    }

    // Decode and dispatch message (processes data in-place before drain)
    DecodeAndDispatch(conn, raw_data, datasize);

    // Drain after processing is complete
    ret = conn->DrainData(datasize);
    if (ret < 0) {
      LOG_ERROR << "DrainData err";
      return;
    }
  }
}

void ProxyCliImpl::DecodeAndDispatch(const ConnectionUeventPtr& conn,
                                     const void* payload, size_t size) {
  // Step 1: Decode proto message header to get message type
  int32_t message_type = GetMessageType(payload, size);

  if (message_type < 0) {
    LOG_FATAL << "Failed to decode message type";
    return;
  }

  // Step 2: Dispatch to appropriate handler based on message type
  switch (message_type) {
    case morphling::global_api::DEVICE_REGISTER_REQUEST:
      HandleRegisterRequest(conn, payload, size);
      break;

    case morphling::global_api::COMPUTE_GEMM_DATA:
      HandleMatMulRequest(conn, payload, size);
      break;

    default:
      LOG_FATAL << "Unknown message type: " << message_type;
      break;
  }
}

void ProxyCliImpl::HandleRegisterRequest(const ConnectionUeventPtr& conn,
                                         const void* payload, size_t size) {
  LOG_DEBUG << "Received registration request from server, size=" << size;

  // Use standard Deserialize interface
  DeviceRegisterRequest request;
  request.Deserialize(payload, size);

  // Request is empty, just send response with device profile
  SendRegisterResponse(conn);
}

void ProxyCliImpl::SendRegisterResponse(const ConnectionUeventPtr& conn) {
  LOG_DEBUG << "Sending device profile data to server";

  DeviceProfileData profile;
  profile.uuid = GenUUID64();
  profile.flops = 100000000000ull;              // 100 GFLOPS - placeholder
  profile.memory = 16ull * 1024 * 1024 * 1024;  // 16GB - placeholder
  profile.ul_bw = 10ull * 1024 * 1024 * 1024;   // 10 Gbps
  profile.dl_bw = 10ull * 1024 * 1024 * 1024;   // 10 Gbps
  profile.ul_lat = 1000;                        // 1ms
  profile.dl_lat = 1000;                        // 1ms

  auto buffer = profile.Serialize();
  // Zero-copy send: buffer ref-count prevents deallocation until libevent done
  auto* ref = new SerializationBufferPtr(buffer);
  int ret = conn->SendDataZeroCopy(buffer->GetBuffer(), buffer->GetSize(),
                                    SerializationBufferSendCleanup, ref);
  if (ret < 0) {
    LOG_ERROR << "Failed to send device profile data";
    conn->ForceClose();
    return;
  }

  LOG_DEBUG << "Device profile data sent, size=" << buffer->GetSize()
            << ", profile: " << profile.DebugString() << ", hex: "
            << BinaryToHex(
                   static_cast<const unsigned char*>(buffer->GetBuffer()),
                   buffer->GetSize())
            << "";
}

void ProxyCliImpl::HandleMatMulRequest(const ConnectionUeventPtr& conn,
                                       const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();

  // Use standard Deserialize interface
  MatrixPartition partition;
  partition.Deserialize(payload, size);

  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " REQ Deserialization time: " << duration.count() << "us";

  // Record DOWNLOAD start time (virtual time)
  uint64_t vt_download_start = VirtualClockNow();
  DEVICE_TRACKER.LogVirtualTimeEvent(partition.dev_id, partition.gemm_id,
                                     "DOWNLOAD", "START", vt_download_start,
                                     vt_download_start);

  // Record bytes received (download request from server)
  DEVICE_TRACKER.RecordBytesReceived(partition.dev_id, size);

  // Log download throughput after receiving request
  double download_tp = DEVICE_TRACKER.GetDownloadThroughput(partition.dev_id);
  double last_packet_tp =
      DEVICE_TRACKER.GetLastPacketThroughput(partition.dev_id);
  double avg_packet_tp =
      DEVICE_TRACKER.GetAveragePacketThroughput(partition.dev_id);
  double server_tp = DEVICE_TRACKER.GetServerAggregatedThroughput();

  uint64_t start_us, end_us;
  DEVICE_TRACKER.GetLastPacketEpochTimestamps(partition.dev_id, start_us,
                                              end_us);

  LOG_INFO << DEV_TAG(device_id_, partition.gemm_id)
           << "[HandleMatMulRequest] Received: " << size << " bytes"
           << " [" << start_us << " -> " << end_us << " us]"
           << ", Download TP: " << download_tp << " B/s"
           << ", Last Packet TP: " << last_packet_tp << " B/s"
           << ", Avg Packet TP: " << avg_packet_tp << " B/s"
           << " | Server Total TP: " << server_tp << " B/s";

  // Log throughput to file
  DEVICE_TRACKER.LogThroughputToFile(partition.dev_id, partition.gemm_id,
                                     "DOWNLOAD", size, download_tp, start_us,
                                     end_us);

  // Process the partition
  HandleMatMul(conn, partition);
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id)
            << "Processed partition: " << partition.DebugString();
}

void ProxyCliImpl::HandleMatMul(const ConnectionUeventPtr& conn,
                                MatrixPartition& partition) {
  auto part_key = partition.GetPartitionKey();
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " partition: " << partition.DebugString();

  // create tensor from partition
  auto [r_ptr, r_size] = partition.mat[0];
  auto [c_ptr, c_size] = partition.mat[1];
  int64_t row_size = r_size / partition.h_dim / sizeof(float);
  int64_t col_size = c_size / partition.h_dim / sizeof(float);

  assert(row_size * partition.h_dim * sizeof(float) == r_size);
  assert(col_size * partition.h_dim * sizeof(float) == c_size);

  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();

  {
    // std::lock_guard<std::mutex> lock(cache_mutex_);
    if (r_size > 0) {
      CacheTensor(tensor_key_row, r_ptr, r_size, partition.h_dim);
    }

    if (c_size > 0) {
      CacheTensor(tensor_key_col, c_ptr, c_size, partition.h_dim);
    }

    auto r_cached = cached_tensors_.Exist(tensor_key_row);
    auto c_cached = cached_tensors_.Exist(tensor_key_col);

    LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
              << " Row cached: " << r_cached << ", row size: " << row_size
              << ", Col cached: " << c_cached << ", col size: " << col_size;

    FillPartition(partition);
    CheckCachedPartition(conn);

    if (r_size == 0 && !r_cached) {
      LOG_WARN << DEV_TAG(device_id_, partition.gemm_id) << part_key
               << " Row not cached, saving for next msg";
      SavePartition(partition);
      return;
    }

    if (c_size == 0 && !c_cached) {
      LOG_WARN << DEV_TAG(device_id_, partition.gemm_id) << part_key
               << " Col not cached, saving for next msg";
      SavePartition(partition);
      return;
    }
  }

  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " Handle partition immediately";
  HandlePartition(conn, partition);
}

void ProxyCliImpl::CheckCachedPartition(
    const uevent::ConnectionUeventPtr& conn) {
  std::vector<std::string> keys;
  for (auto& c_part : cached_partitions_) {
    FillPartition(c_part);
    auto r_size = std::get<1>(c_part.mat[0]);
    auto c_size = std::get<1>(c_part.mat[1]);
    if (r_size > 0 && c_size > 0) {
      auto key = c_part.GetPartitionKey();
      LOG_DEBUG << DEV_TAG(c_part.dev_id, c_part.gemm_id) << key
                << " Handle partition from cache";
      HandlePartition(conn, c_part);
      keys.push_back(key);
    } else {
      // LOG_WARN << c_part.GetPartitionKey()
      //          << " Partition is not ready, r_size: " << r_size
      //          << ", c_size: " << c_size;
    }
  }

  for (auto it = cached_partitions_.begin(); it != cached_partitions_.end();) {
    if (std::find(keys.begin(), keys.end(), it->GetPartitionKey()) !=
        keys.end()) {
      it = cached_partitions_.erase(it);
    } else {
      ++it;
    }
  }
}

void ProxyCliImpl::HandlePartition(const ConnectionUeventPtr& conn,
                                   const MatrixPartition& partition) {
  auto* loop = conn->GetLoop();
  loop->AssertInLoopThread();

  auto* loop_handle = loop->GetLoopHandle();
  auto* handle = reinterpret_cast<ProxyCliHandle*>(loop_handle);
  // handle->HandlePartition(conn, partition);
  // Copy partition by value for safety with async dispatch
  MatrixPartition part_copy = partition;
  loop->RunInLoop(bind(&ProxyCliHandle::HandlePartition, handle, conn,
                       std::move(part_copy)));
}

MatrixPartition ProxyCliImpl::DecodeRequest(const void* payload, size_t size) {
  auto start = std::chrono::high_resolution_clock::now();
  MatrixPartition partition;
  partition.Deserialize(payload, size, SerializationFormat::PROTOBUF);
  auto part_key = partition.GetPartitionKey();
  auto end = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id) << part_key
            << " REQ Deserialization time: " << duration.count() << "us";
  return partition;
}

void ProxyCliImpl::CacheTensor(const TensorKey& key, void* ptr, int64_t size,
                               int64_t h_dim) {
  LOG_INFO << DEV_TAG_DEV(device_id_) << "CacheTensor called: ptr=" << ptr
           << ", size=" << size << ", h_dim=" << h_dim;

  if (cached_tensors_.Exist(key)) {
    LOG_DEBUG << DEV_TAG_DEV(device_id_) << "Tensor already cached, skipping";
    return;
  }

  if (ptr == nullptr) {
    LOG_ERROR << DEV_TAG_DEV(device_id_) << "CacheTensor: ptr is nullptr!";
    return;
  }

  if (size <= 0) {
    LOG_ERROR << DEV_TAG_DEV(device_id_)
              << "CacheTensor: invalid size=" << size;
    return;
  }

  if (h_dim <= 0) {
    LOG_ERROR << "CacheTensor: invalid h_dim=" << h_dim;
    return;
  }

  LOG_DEBUG << "Allocating managed memory: size=" << size << " bytes";
  void* cpy_ptr = nullptr;
#ifdef CACHEDTENSOR_CUDA_MALLOC_MANAGED
  if (!LogCudaError(cudaMallocManaged(&cpy_ptr, size),
                    "CacheTensor cudaMallocManaged")) {
    return;
  }
  LOG_DEBUG << "cudaMallocManaged succeeded: cpy_ptr=" << cpy_ptr;
#else
  cpy_ptr = malloc(size);
  if (cpy_ptr == nullptr) {
    LOG_ERROR << "CacheTensor: malloc failed for size=" << size;
    return;
  }
#endif

  int64_t ld_size = size / h_dim / sizeof(float);
  LOG_DEBUG << "Calculated ld_size=" << ld_size << " (size=" << size
            << " / h_dim=" << h_dim << " / sizeof(float)=" << sizeof(float)
            << ")";

  LOG_DEBUG << "Copying data: from " << ptr << " to " << cpy_ptr
            << ", size=" << size << " bytes";
#ifdef CACHEDTENSOR_CUDA_MALLOC_MANAGED
  cudaMemcpy(cpy_ptr, ptr, size, cudaMemcpyDefault);
#else
  std::memcpy(cpy_ptr, ptr, size);
#endif

  LOG_DEBUG << DEV_TAG_DEV(device_id_) << "memcpy completed successfully";

  cached_tensors_.Put(key, CachedTensor{cpy_ptr, ld_size, h_dim, size}, size);

  LOG_INFO << "CacheTensor completed successfully";
}

void ProxyCliImpl::FillPartition(MatrixPartition& partition) {
  auto r_size = std::get<1>(partition.mat[0]);
  auto c_size = std::get<1>(partition.mat[1]);
  auto tensor_key_row = partition.GetRowKey();
  auto tensor_key_col = partition.GetColKey();
  auto r_cached = cached_tensors_.Exist(tensor_key_row);
  auto c_cached = cached_tensors_.Exist(tensor_key_col);

  if (!r_cached || !c_cached) {
    LOG_DEBUG << DEV_TAG(device_id_, partition.gemm_id)
              << partition.GetPartitionKey()
              << " FillPartition: r_cached=" << r_cached
              << ", c_cached=" << c_cached << ", r_size=" << r_size
              << ", c_size=" << c_size;
  }

  if (r_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_row);
    partition.mat[0] = {cached_tensor.data,
                        static_cast<int64_t>(cached_tensor.bytes)};
  }
  if (c_cached) {
    auto cached_tensor = cached_tensors_.Get(tensor_key_col);
    partition.mat[1] = {cached_tensor.data,
                        static_cast<int64_t>(cached_tensor.bytes)};
  }
  // if (r_size == 0 && r_cached) {
  //   auto cached_tensor = cached_tensors_.Get(tensor_key_row);
  //   partition.mat[0] = {cached_tensor.data_ptr(),
  //                       cached_tensor.numel() * sizeof(float)};
  // }

  // if (c_size == 0 && c_cached) {
  //   auto cached_tensor = cached_tensors_.Get(tensor_key_col);
  //   partition.mat[1] = {cached_tensor.data_ptr(),
  //                       cached_tensor.numel() * sizeof(float)};
  // }
}

void ProxyCliImpl::SavePartition(MatrixPartition& partition) {
  for (auto& mat : partition.mat) {
    mat = {nullptr, 0};
  }
  cached_partitions_.push_back(partition);
}

/*********************************ProxyCli***************************************/
typedef ProxyCli::Status ProxyStatus;
typedef ProxyStatus::StatusType ProxyStatusType;
const map<ProxyStatusType, string> ProxyStatus::status_str_ = {
    {kOK, ""},
    {kFatal, "Proxy svr fatal:"},
    {kUnknown, "Proxy svr unknown:"},
};

ProxyCli::ProxyCli() : svr_(nullptr), loop_thread_(nullptr) {}

ProxyCli::~ProxyCli() = default;

void ProxyCli::Initialize(const std::string& cfg_file, int64_t device_id) {
  // detect cuda availability and set device accordingly
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  bool has_gpu = (err == cudaSuccess && device_count > 0);

  if (!has_gpu) {
    if (err != cudaSuccess) {
      LOG_WARN << "Failed to get CUDA device count: "
               << cudaGetErrorString(err) << ". Running in CPU-only mode.";
    } else {
      LOG_WARN << "No CUDA devices found. Running in CPU-only mode.";
    }
    AlignedBufferPool::instance().SetPinFunctions(PosixPinBuffer,
                                                  PosixUnpinBuffer);
  } else {
    LOG_INFO << "CUDA devices detected: " << device_count
             << ". Running in GPU mode.";
    AlignedBufferPool::instance().SetPinFunctions(CudaPinBuffer,
                                                  CudaUnpinBuffer);
  }

  context_.Initialize(cfg_file);

  bool want_gpu = (context_.pool_mode == "gpu" ||
                   context_.pool_mode == "both");
  bool want_cpu = (context_.pool_mode == "cpu" ||
                   context_.pool_mode == "both");

  // Create worker pools based on config + hardware availability
  if (want_gpu && has_gpu) {
    int workers_per_gpu = 1;  // one green-context partition per GPU
    size_t buffer_size = 256ull * 1024 * 1024;  // 256 MB per worker
    gpu_pool_ = std::make_unique<XtGemmWorkerPool>(
        workers_per_gpu, buffer_size,
        WorkerSchedulingPolicy::kRoundRobinGemm);
    LOG_INFO << "GPU worker pool created: " << workers_per_gpu
             << " workers/GPU, " << device_count << " GPUs";
  }
  if (want_cpu) {
    int num_cores = static_cast<int>(std::thread::hardware_concurrency());
    int num_workers = std::max(1, num_cores / 2);
    std::vector<int> cores;
    cores.reserve(num_workers);
    for (int i = 0; i < num_workers; i++) {
      cores.push_back(i);
    }
    cpu_pool_ = std::make_unique<CpuWorkerPool>(
        num_workers, std::move(cores),
        WorkerSchedulingPolicy::kRoundRobinCpu);
    LOG_INFO << "CPU worker pool created: " << num_workers << " workers";
  }
  svr_ = make_shared<ProxyCliImpl>(context_, device_id);
  loop_thread_ = make_shared<UeventLoopThread>(
      bind(ProxyCliHandle::CreateMyself, ref(context_), device_id,
           gpu_pool_.get(), cpu_pool_.get(), _1),
      bind(&ProxyCliImpl::Initialize, svr_, _1), "Proxy svr main thread");
}
void ProxyCli::Start() { loop_thread_->StartLoop(); }
void ProxyCli::Send(const torch::Tensor& tensor, std::optional<int64_t> rank) {}
void ProxyCli::Receive(torch::Tensor& tensor, std::optional<int64_t> rank) {}
void ProxyCli::AsyncSend(const torch::Tensor& tensor,
                         std::optional<int64_t> rank) {}
void ProxyCli::AsyncReceive(torch::Tensor& tensor,
                            std::optional<int64_t> rank) {}

}  // namespace backend
}  // namespace morphling
