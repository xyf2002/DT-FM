#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <map>
#include <memory>
#include <utility>

#include "base/logging.h"
#include "base/thread.h"
#include "network/connector_libevent.h"
#include "network/eventloop_libevent.h"
#include "network/uevent.h"

using std::placeholders::_1;
using std::placeholders::_2;
using namespace uevent;
using namespace base;

class EchoConnector;

typedef std::shared_ptr<EchoConnector> EchoConnectorPtr;
std::vector<EchoConnectorPtr> connectors;

static const uint32_t kMaxPacketLen = 1024 * 1024 * 4;
static const uint32_t kPacketLen = kMaxPacketLen + sizeof(uint32_t);
static const void* msg = malloc(kPacketLen);

class EchoConnector {
 public:
  EchoConnector(UeventLoop* loop, const UsockAddress& listenAddr,
                const std::string& id)
      : loop_(loop), connector_(loop, listenAddr, "EchoConnector" + id) {
    connector_.SetConnectionSuccessCb(
        std::bind(&EchoConnector::ConnectionSuccessHandle, this, _1));
    connector_.SetConnectionClosedCb(
        std::bind(&EchoConnector::ConnectionClosedHandle, this, _1));
    connector_.SetMessageReadCb(
        std::bind(&EchoConnector::MessageReadHandle, this, _1));
    connector_.SetMessageWriteCb(
        std::bind(&EchoConnector::MessageWriteHandle, this, _1));
  }

  void Connect() {
    connector_.Connect();
    uint32_t* len = (uint32_t*)msg;
    *len = htonl(kMaxPacketLen);
  }

 private:
  size_t total_bytes_received_ = 0;
  size_t total_messages_received_ = 0;
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;

  void ConnectionSuccessHandle(const ConnectionUeventPtr& conn) {
    LOG_INFO << "connect to " << conn->GetPeerAddress().ToString()
             << " success";
    start_time_ = std::chrono::high_resolution_clock::now();
    conn->SendData(msg, kPacketLen);
  }

  void ConnectionClosedHandle(const ConnectionUeventPtr& conn) {
    LOG_INFO << "connect to " << conn->GetPeerAddress().ToString() << " failed";
    end_time_ = std::chrono::high_resolution_clock::now();
    ReportBenchmark();
    connectors.clear();  // 验证析构connector时内部链接是否正确释放
  }

  void MessageReadHandle(const ConnectionUeventPtr& conn) {
    while (true) {
      size_t readable = conn->ReadableLength();

      int ret = 0;
      size_t packsize;
      ret = conn->ReceiveData(&packsize, sizeof(size_t));
      if (ret < 0) {
        LOG_ERROR << "ReceiveData packsize err";
        return;
      }
      packsize = ntohl(packsize);
      uint32_t datasize = packsize + sizeof(uint32_t);

      if (static_cast<uint32_t>(readable) < datasize) {
        return;
      }
      std::unique_ptr<char[]> data(new char[datasize]);
      char* raw_data = data.get();
      ret = conn->ReceiveData(raw_data, datasize);
      if (ret < 0) {
        LOG_ERROR << "ReceiveData raw_data err";
        return;
      }

      ret = conn->DrainData(datasize);
      if (ret < 0) {
        LOG_ERROR << "DrainData err";
        return;
      }

      total_bytes_received_ += datasize;
      total_messages_received_++;
      // LOG_INFO << "connection id: " << conn->GetId() << " receive " <<
      // readable << " bytes";
      if (total_bytes_received_ >= 1024 * 1024 * 1024) {
        LOG_INFO << "Received 1 GB of data, closing connection.";
        conn->ForceClose();
      }
    }
  }

  void MessageWriteHandle(const ConnectionUeventPtr& conn) {
    // std::string msg = "wo shi ye heng";
    conn->SendData(msg, kPacketLen);
  }

  void ReportBenchmark() {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                        end_time_ - start_time_)
                        .count();
    double throughput_mb = static_cast<double>(total_bytes_received_) /
                           (1024 * 1024) / duration * 1e6;
    double messages_per_second =
        static_cast<double>(total_messages_received_) / duration * 1e6;

    LOG_INFO << "Benchmark Results:";
    LOG_INFO << "Throughput: " << throughput_mb << " MB/s";
    LOG_INFO << "Messages per second: " << messages_per_second << " msg/s";
  }

  UeventLoop* loop_;
  ConnectorLibevent connector_;
};

int main(int argc, char* argv[]) {
  LOG_INFO << "pid = " << getpid() << ", tid = " << CurrentThread::tid();
  if (argc > 2) {
    int port = atoi(argv[2]);
    EventLoopLibevent loop("main_thread");
    UsockAddress server_addr(argv[1], port, false);
    int n = 1;
    if (argc > 3) {
      n = atoi(argv[3]);
    }
    for (int i = 0; i < n; ++i) {
      char buf[32];
      snprintf(buf, sizeof buf, "%d", i + 1);
      EchoConnectorPtr ctor =
          std::make_shared<EchoConnector>(&loop, server_addr, buf);
      ctor->Connect();
      connectors.push_back(ctor);
    }
    loop.Start();
  } else {
    printf("Usage: %s host_ip port [current#]\n", argv[0]);
  }
}
