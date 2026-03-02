#ifndef UEVENT_USOCK_ADDRESS_H
#define UEVENT_USOCK_ADDRESS_H

#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>

#include <string>

namespace uevent {

class UsockAddress {
 public:
  /// Constructs an endpoint with given ip and port.
  /// @c ip should be "1.2.3.4"
  UsockAddress(const std::string& ip, uint16_t port, bool ipv6 = false);

  UsockAddress(const std::string& path);
  /// Constructs an endpoint with given struct @c sockaddr_in
  /// Mostly used when accepting new connections
  explicit UsockAddress(const struct sockaddr_in& addr) : addr_(addr) {}

  explicit UsockAddress(const struct sockaddr_in6& addr) : addr6_(addr) {}
  explicit UsockAddress(const struct sockaddr_un& addr) : un_addr_(addr) {}

  explicit UsockAddress(const struct sockaddr_storage& addr)
      : gen_addr_(addr) {}

  sa_family_t family() const { return addr_.sin_family; }
  std::string ToIpString() const;
  std::string ToIpPortString() const;
  uint16_t ToPort() const;
  std::string ToString() const;

  // default copy/assignment are Okay

  const struct sockaddr* GetSockAddr() const {
    return reinterpret_cast<const sockaddr*>(&gen_addr_);
  }
  int GetSockAddrLen();

  struct sockaddr* GetSockAddr() {
    return reinterpret_cast<sockaddr*>(&gen_addr_);
  }
  void SetSockAddrInet6(const struct sockaddr_in6& addr6) { addr6_ = addr6; }
  void SetSockAddrGen(const struct sockaddr_storage& addr) { gen_addr_ = addr; }

  uint32_t IpNetEndian() const;
  uint16_t PortNetEndian() const { return addr_.sin_port; }

  // resolve hostname to IP address, not changing port or sin_family
  // return true on success.
  // thread safe
  static bool Resolve(const std::string& hostname, UsockAddress* result);
  // static std::vector<InetAddress> resolveAll(const char* hostname, uint16_t
  // port = 0);
  static void FromIpPort(const char* ip, uint16_t port,
                         struct sockaddr_in* addr);

  static void FromIpPort(const char* ip, uint16_t port,
                         struct sockaddr_in6* addr);

  static void FromUnPath(const char* path, struct sockaddr_un* addr);

  static void ToIpPortUtil(char* buf, size_t size, const struct sockaddr* addr);

  static void ToIpUtil(char* buf, size_t size, const struct sockaddr* addr);

 private:
  static const struct sockaddr* sockaddr_cast(const struct sockaddr_in6* addr) {
    return reinterpret_cast<const struct sockaddr*>(addr);
  }

  static struct sockaddr* sockaddr_cast(struct sockaddr_in6* addr) {
    return reinterpret_cast<struct sockaddr*>(addr);
  }

  static const struct sockaddr* sockaddr_cast(const struct sockaddr_in* addr) {
    return reinterpret_cast<const struct sockaddr*>(addr);
  }

  static const struct sockaddr_in* sockaddr_in_cast(
      const struct sockaddr* addr) {
    return reinterpret_cast<const struct sockaddr_in*>(addr);
  }

  static const struct sockaddr_in6* sockaddr_in6_cast(
      const struct sockaddr* addr) {
    return reinterpret_cast<const struct sockaddr_in6*>(addr);
  }

  union {
    struct sockaddr_in addr_;
    struct sockaddr_in6 addr6_;
    struct sockaddr_un un_addr_;
    struct sockaddr_storage gen_addr_;
  };
};
}  // namespace uevent

#endif  // UEVENT_USOCK_ADDRESS_H
