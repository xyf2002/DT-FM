#include <sys/mman.h>
#include <unistd.h>
#include <uv.h>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>

using namespace std::chrono;

class EchoClient {
 public:
  EchoClient(uv_loop_t* loop, const char* ip, int port)
      : loop_(loop), message_(GenerateRandomMessage(4 * 1024 * 1024)) {
    int r = uv_tcp_init(loop_, &client_);
    if (r != 0) {
      std::cerr << "Failed to initialize TCP handle: " << uv_strerror(r)
                << std::endl;
      exit(1);
    }
    fprintf(stdout, "uv_tcp_init success\n");

    r = uv_ip4_addr(ip, port, &server_addr_);
    if (r != 0) {
      std::cerr << "Failed to initialize server address: " << uv_strerror(r)
                << std::endl;
      exit(1);
    }
    fprintf(stdout, "uv_ip4_addr success\n");

    uint32_t packet_len = htonl(message_.size());
    std::string packet(reinterpret_cast<char*>(&packet_len), sizeof(uint32_t));
    packet += message_;
    message_ = packet;
  }

  void Start() {
    int r = uv_tcp_connect(&connect_req_, &client_,
                           (const struct sockaddr*)&server_addr_, OnConnect);
    if (r != 0) {
      std::cerr << "Failed to connect: " << uv_strerror(r) << std::endl;
      exit(1);
    }
    fprintf(stdout, "uv_tcp_connect success\n");
    connect_req_.data = this;
  }

 private:
  uv_loop_t* loop_;
  uv_tcp_t client_;
  uv_connect_t connect_req_;
  struct sockaddr_in server_addr_;
  std::string message_;
  std::vector<char> buffer_;

  size_t total_bytes_received_ = 0;
  size_t total_messages_received_ = 0;
  high_resolution_clock::time_point start_time_;
  high_resolution_clock::time_point end_time_;

  static void OnConnect(uv_connect_t* req, int status) {
    if (status < 0) {
      std::cerr << "Connection error: " << uv_strerror(status) << std::endl;
      return;
    }

    auto* self = static_cast<EchoClient*>(req->data);
    self->start_time_ = high_resolution_clock::now();

    uv_buf_t buf = uv_buf_init(const_cast<char*>(self->message_.data()),
                               self->message_.size());
    uv_write_t* write_req = new uv_write_t;
    write_req->data = self;
    uv_write(write_req, req->handle, &buf, 1, OnWrite);

    uv_read_start(req->handle, AllocBuffer, OnRead);
    std::cout << "Connected to server, sending message..." << std::endl;
  }

  static void AllocBuffer(uv_handle_t* handle, size_t suggested_size,
                          uv_buf_t* buf) {
    buf->base = (char*)malloc(suggested_size);
    buf->len = suggested_size;
    // auto self = static_cast<EchoClient*>(handle->data);
    // assert(suggested_size > self->buffer_size_);
    // if (self->buffer_offset_ + suggested_size > self->buffer_size_) {
    //   self->buffer_offset_ = 0;
    // }
    // buf->base = static_cast<char*>(self->buffer_ + self->buffer_offset_);
    // buf->len = suggested_size;
    // self->buffer_offset_ = (self->buffer_offset_ + suggested_size) %
    // self->buffer_size_;
  }

  static void OnRead(uv_stream_t* client, ssize_t nread, const uv_buf_t* buf) {
    auto* self = static_cast<EchoClient*>(client->data);

    if (nread > 0) {
      self->buffer_.insert(self->buffer_.end(), buf->base, buf->base + nread);

      while (self->buffer_.size() >= sizeof(uint32_t)) {
        uint32_t packet_len;
        std::memcpy(&packet_len, self->buffer_.data(), sizeof(uint32_t));
        packet_len = ntohl(packet_len);

        if (self->buffer_.size() < sizeof(uint32_t) + packet_len) {
          break;
        }
        std::cout << "Buffer size: " << self->buffer_.size() << std::endl;
        std::cout << "Packet length: " << packet_len << std::endl;
        std::string data(self->buffer_.begin() + sizeof(uint32_t),
                         self->buffer_.begin() + sizeof(uint32_t) + packet_len);
        self->buffer_.erase(
            self->buffer_.begin(),
            self->buffer_.begin() + sizeof(uint32_t) + packet_len);

        std::cout << "Received echo: " << data.size() << " bytes" << std::endl;
      }
    } else if (nread < 0) {
      if (nread != UV_EOF) {
        std::cerr << "Read error: " << uv_strerror(nread) << std::endl;
      }
      uv_close((uv_handle_t*)client, OnClose);
    }
    // free(buf->base);
  }

  static void OnWrite(uv_write_t* req, int status) {
    if (status < 0) {
      std::cerr << "Write error: " << uv_strerror(status) << std::endl;
    }

    auto* self = static_cast<EchoClient*>(req->data);
    delete req;

    uv_buf_t buf = uv_buf_init(const_cast<char*>(self->message_.data()),
                               self->message_.size());
    uv_write_t* write_req = new uv_write_t;
    write_req->data = self;
    uv_write(write_req, write_req->handle, &buf, 1, OnWrite);
  }

  static void OnClose(uv_handle_t* handle) {
    auto* self = static_cast<EchoClient*>(handle->data);
    self->end_time_ = high_resolution_clock::now();
    self->ReportBenchmark();
    delete handle;
  }

  void ReportBenchmark() {
    auto duration =
        duration_cast<microseconds>(end_time_ - start_time_).count();
    double throughput_mb = static_cast<double>(total_bytes_received_) /
                           (1024 * 1024) / duration * 1e6;
    double messages_per_second =
        static_cast<double>(total_messages_received_) / duration * 1e6;

    std::cout << "Benchmark Results:" << std::endl;
    std::cout << "Throughput: " << throughput_mb << " MB/s" << std::endl;
    std::cout << "Messages per second: " << messages_per_second << " msg/s"
              << std::endl;
  }

  static std::string GenerateRandomMessage(size_t size) {
    std::string message(size, '\0');
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<> dis(0, 255);

    // for (size_t i = 0; i < size; ++i) {
    //   message[i] = static_cast<char>(dis(gen));
    // }
    return message;
  }
};

int main(int argc, char* argv[]) {
  fprintf(stdout, "argv[1]:%s;argv[2]:%s\n", argv[1], argv[2]);
  const char* ip = argv[1];
  int port = std::stoi(argv[2]);
  fprintf(stdout, "ip:%s;port:%d\n", ip, port);

  uv_loop_t* loop = uv_default_loop();
  EchoClient client(loop, ip, port);
  client.Start();

  return uv_run(loop, UV_RUN_DEFAULT);
}
