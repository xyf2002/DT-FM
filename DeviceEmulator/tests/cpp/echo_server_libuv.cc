#include <uv.h>

#include <cstring>
#include <iostream>
#include <vector>

class EchoServer {
 public:
  EchoServer(uv_loop_t* loop, const char* ip, int port) : loop_(loop) {
    uv_tcp_init(loop_, &server_);
    uv_ip4_addr(ip, port, &addr_);
  }

  void Start() {
    uv_tcp_bind(&server_, (const struct sockaddr*)&addr_, 0);
    server_.data = this;
    uv_listen((uv_stream_t*)&server_, 128, OnNewConnection);
    std::cout << "Echo server started on port " << addr_.sin_port << std::endl;
  }

 private:
  static void OnNewConnection(uv_stream_t* server, int status) {
    if (status < 0) {
      std::cerr << "New connection error: " << uv_strerror(status) << std::endl;
      return;
    }

    auto* self = static_cast<EchoServer*>(server->data);
    uv_tcp_t* client = new uv_tcp_t;
    uv_tcp_init(self->loop_, client);
    if (uv_accept(server, (uv_stream_t*)client) == 0) {
      client->data = self;
      uv_read_start((uv_stream_t*)client, AllocBuffer, OnRead);
    } else {
      uv_close((uv_handle_t*)client, OnClose);
    }
  }

  static void AllocBuffer(uv_handle_t* handle, size_t suggested_size,
                          uv_buf_t* buf) {
    buf->base = (char*)malloc(suggested_size);
    buf->len = suggested_size;
  }

  static void OnRead(uv_stream_t* client, ssize_t nread, const uv_buf_t* buf) {
    if (nread > 0) {
      auto* self = static_cast<EchoServer*>(client->data);

      self->buffer_.insert(self->buffer_.end(), buf->base, buf->base + nread);

      while (self->buffer_.size() >= sizeof(uint32_t)) {
        uint32_t packet_len;
        std::memcpy(&packet_len, self->buffer_.data(), sizeof(uint32_t));
        packet_len = ntohl(packet_len);

        if (self->buffer_.size() < sizeof(uint32_t) + packet_len) {
          break;
        }

        std::string data(self->buffer_.begin() + sizeof(uint32_t),
                         self->buffer_.begin() + sizeof(uint32_t) + packet_len);
        self->buffer_.erase(
            self->buffer_.begin(),
            self->buffer_.begin() + sizeof(uint32_t) + packet_len);

        uv_buf_t write_buf =
            uv_buf_init(const_cast<char*>(data.data()), data.size());
        uv_write_t* write_req = new uv_write_t;
        uv_write(write_req, client, &write_buf, 1, OnWrite);
      }
    } else if (nread < 0) {
      if (nread != UV_EOF) {
        std::cerr << "Read error: " << uv_strerror(nread) << std::endl;
      }
      uv_close((uv_handle_t*)client, OnClose);
    }
    free(buf->base);
  }

  static void OnWrite(uv_write_t* req, int status) {
    if (status < 0) {
      std::cerr << "Write error: " << uv_strerror(status) << std::endl;
    }
    delete req;
  }

  static void OnClose(uv_handle_t* handle) { delete handle; }

  uv_loop_t* loop_;
  uv_tcp_t server_;
  struct sockaddr_in addr_;
  std::vector<char> buffer_;
};

int main(int argc, char* argv[]) {
  const char* ip = "0.0.0.0";
  int port = 13500;

  uv_loop_t* loop = uv_default_loop();
  EchoServer server(loop, ip, port);
  server.Start();

  return uv_run(loop, UV_RUN_DEFAULT);
}
