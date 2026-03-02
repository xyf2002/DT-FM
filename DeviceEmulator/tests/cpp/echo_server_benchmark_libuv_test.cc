#include <uv.h>

#include <chrono>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

using namespace std::chrono;

class BenchmarkClient {
 public:
  BenchmarkClient(uv_loop_t* loop, const std::string& server_ip,
                  int server_port, size_t message_size, size_t message_count)
      : loop_(loop),
        message_size_(message_size),
        message_count_(message_count),
        messages_sent_(0),
        messages_received_(0) {
    uv_tcp_init(loop_, &client_);
    uv_ip4_addr(server_ip.c_str(), server_port, &server_addr_);
  }

  void Start() {
    uv_tcp_connect(&connect_req_, &client_,
                   (const struct sockaddr*)&server_addr_, OnConnect);
    connect_req_.data = this;
  }

  size_t GetMessagesReceived() const { return messages_received_; }

 private:
  static void OnConnect(uv_connect_t* req, int status) {
    if (status < 0) {
      std::cerr << "Connection error: " << uv_strerror(status) << std::endl;
      return;
    }

    auto* self = static_cast<BenchmarkClient*>(req->data);
    self->SendMessage();

    uv_read_start(req->handle, AllocBuffer, OnRead);
  }

  static void AllocBuffer(uv_handle_t* handle, size_t suggested_size,
                          uv_buf_t* buf) {
    buf->base = (char*)malloc(suggested_size);
    buf->len = suggested_size;
  }

  static void OnRead(uv_stream_t* stream, ssize_t nread, const uv_buf_t* buf) {
    auto* self = static_cast<BenchmarkClient*>(stream->data);

    if (nread > 0) {
      ++self->messages_received_;
      if (self->messages_sent_ < self->message_count_) {
        self->SendMessage();
      } else if (self->messages_received_ == self->message_count_) {
        uv_stop(self->loop_);
      }
    } else if (nread < 0) {
      std::cerr << "Read error: " << uv_strerror(nread) << std::endl;
    }

    free(buf->base);
  }

  void SendMessage() {
    std::string message(message_size_, 'x');
    uv_buf_t buf =
        uv_buf_init(const_cast<char*>(message.data()), message.size());

    uv_write_t* write_req = new uv_write_t;
    write_req->data = this;
    uv_write(write_req, (uv_stream_t*)&client_, &buf, 1, OnWrite);

    ++messages_sent_;
  }

  static void OnWrite(uv_write_t* req, int status) {
    if (status < 0) {
      std::cerr << "Write error: " << uv_strerror(status) << std::endl;
    }
    delete req;
  }

  uv_loop_t* loop_;
  uv_tcp_t client_;
  uv_connect_t connect_req_;
  struct sockaddr_in server_addr_;
  size_t message_size_;
  size_t message_count_;
  size_t messages_sent_;
  size_t messages_received_;
};

int main(int argc, char* argv[]) {
  if (argc < 5) {
    std::cerr << "Usage: " << argv[0]
              << " <server_ip> <server_port> <message_size> <message_count> "
                 "[client_count]"
              << std::endl;
    return 1;
  }

  std::string server_ip = argv[1];
  int server_port = std::stoi(argv[2]);
  size_t message_size = std::stoul(argv[3]);
  size_t message_count = std::stoul(argv[4]);
  size_t client_count = (argc > 5) ? std::stoul(argv[5]) : 1;

  uv_loop_t* loop = uv_default_loop();

  std::vector<std::unique_ptr<BenchmarkClient>> clients;
  for (size_t i = 0; i < client_count; ++i) {
    clients.emplace_back(std::make_unique<BenchmarkClient>(
        loop, server_ip, server_port, message_size, message_count));
    clients.back()->Start();
  }

  auto start_time = high_resolution_clock::now();
  uv_run(loop, UV_RUN_DEFAULT);
  auto end_time = high_resolution_clock::now();

  size_t total_messages_received = 0;
  for (const auto& client : clients) {
    total_messages_received += client->GetMessagesReceived();
  }

  auto duration = duration_cast<milliseconds>(end_time - start_time).count();
  double throughput =
      static_cast<double>(total_messages_received) / (duration / 1000.0);

  std::cout << "Benchmark Results:" << std::endl;
  std::cout << "Total Messages Received: " << total_messages_received
            << std::endl;
  std::cout << "Total Time (ms): " << duration << std::endl;
  std::cout << "Throughput (messages/sec): " << throughput << std::endl;

  return 0;
}
