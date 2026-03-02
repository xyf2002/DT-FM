#include <stdio.h>
#include <unistd.h>

#include <memory>
#include <string>
#include <utility>

#include "base/logging.h"
#include "base/thread.h"
#include "network/eventloop_libevent.h"
#include "network/listener_libevent.h"
#include "network/loop_handle.h"
#include "network/uevent.h"

using namespace uevent;
using namespace base;

std::unique_ptr<ListenerUevent> g_listener;
int numThreads = 10;

class EchoServer : public LoopHandle {
 public:
  static LoopHandle* CreateMyself(UeventLoop* loop) {
    return reinterpret_cast<LoopHandle*>(new EchoServer(loop));
  }
  EchoServer(UeventLoop* loop) : name_("henry server") {}
  static void ConnectionSuccessHandle(const ConnectionUeventPtr& conn) {
    LoopHandle* loop_handle = conn->GetLoop()->GetLoopHandle();
    EchoServer* pobj = reinterpret_cast<EchoServer*>(loop_handle);
    LOG_INFO << "echo sever name: " << pobj->name_;
    LOG_INFO << "connection from " << conn->GetPeerAddress().ToString()
             << " success";
  }
  static void ConnectionClosedHandle(const ConnectionUeventPtr& conn) {
    // LoopHandle* loop_handle = conn->GetLoop()->GetLoopHandle();
    // EchoServer* pobj = reinterpret_cast<EchoServer*>(loop_handle);
    LOG_INFO << "disconnect from " << conn->GetPeerAddress().ToString();
    g_listener->RemoveConnection(conn);  // 验证连接的释放
  }

  static void MessageReadHandle(const ConnectionUeventPtr& conn) {
    LoopHandle* loop_handle = conn->GetLoop()->GetLoopHandle();
    EchoServer* pobj = reinterpret_cast<EchoServer*>(loop_handle);
    // LOG_INFO << "echo sever name: " << pobj->name_;

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

      // void* data = malloc(readable);
      // conn->ReceiveData(data, readable);
      // conn->RemoveData(data, readable);
      LOG_TRACE << "Connection Id: " << conn->GetId() << " recv " << datasize
                << " bytes";
      conn->SendData(raw_data, datasize);
    }
  }
  static void MessageWriteHandle(const ConnectionUeventPtr& conn) {
    LOG_TRACE << "MessageWriteHandle";
  }

 private:
  const std::string name_;
  UeventLoop* loop_;
  ListenerUevent* listener_;
};

int main(int argc, char* argv[]) {
  Option option;
  option.loop_strategy = Option::kRoundRobin;
  option.reuse_port = true;
  LOG_INFO << "pid = " << getpid() << ", tid = " << CurrentThread::tid();
  LOG_INFO << "sizeof ConnectionUevent = " << sizeof(ConnectionUevent);
  if (argc > 1) {
    numThreads = atoi(argv[1]);
  }
  EventLoopLibevent loop("main_thread", EchoServer::CreateMyself);
  UsockAddress listenAddr("0.0.0.0", 13500, false);
  // unlink("/root/yeheng.sock");
  // UsockAddress listenAddr("/root/yeheng.sock");

  g_listener.reset(
      new ListenerLibevent(&loop, listenAddr, "EchoListener", option));
  g_listener->SetConnectionSuccessCb(EchoServer::ConnectionSuccessHandle);
  g_listener->SetConnectionClosedCb(EchoServer::ConnectionClosedHandle);
  g_listener->SetMessageReadCb(EchoServer::MessageReadHandle);
  g_listener->SetMessageWriteCb(EchoServer::MessageWriteHandle);
  g_listener->SetCreateLoopHandleCb(EchoServer::CreateMyself);
  g_listener->SetThreadNum(numThreads);

  g_listener->Start();
}
