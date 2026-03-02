#include "usock_address.h"

#include <arpa/inet.h>
#include <netdb.h>

#include "base/logging.h"

//     /* Structure describing an Internet socket address.  */
//     struct sockaddr_in {
//         sa_family_t    sin_family; /* address family: AF_INET */
//         uint16_t       sin_port;   /* port in network byte order */
//         struct in_addr sin_addr;   /* internet address */
//     };

//     /* Internet address. */
//     typedef uint32_t in_addr_t;
//     struct in_addr {
//         in_addr_t       s_addr;     /* address in network byte order */
//     };

//     struct sockaddr_in6 {
//         sa_family_t     sin6_family;   /* address family: AF_INET6 */
//         uint16_t        sin6_port;     /* port in network byte order */
//         uint32_t        sin6_flowinfo; /* IPv6 flow information */
//         struct in6_addr sin6_addr;     /* IPv6 address */
//         uint32_t        sin6_scope_id; /* IPv6 scope-id */
//     };

namespace uevent {

UsockAddress::UsockAddress(const std::string& ip, uint16_t port, bool ipv6) {
  if (ipv6) {
    bzero(&addr6_, sizeof addr6_);
    FromIpPort(ip.c_str(), port, &addr6_);
  } else {
    bzero(&addr_, sizeof addr_);
    FromIpPort(ip.c_str(), port, &addr_);
  }
}

UsockAddress::UsockAddress(const std::string& path) {
  bzero(&un_addr_, sizeof un_addr_);
  FromUnPath(path.c_str(), &un_addr_);
}

std::string UsockAddress::ToIpPortString() const {
  char buf[64] = "";
  ToIpPortUtil(buf, sizeof buf, GetSockAddr());
  return buf;
}

std::string UsockAddress::ToIpString() const {
  char buf[64] = "";
  ToIpUtil(buf, sizeof buf, GetSockAddr());
  return buf;
}

std::string UsockAddress::ToString() const {
  switch (gen_addr_.ss_family) {
    case AF_LOCAL:
      return std::string(un_addr_.sun_path);
    case AF_INET:
    case AF_INET6:
      return ToIpPortString();
    default:
      LOG_SYSERR << "UsockAddress::ToString";
      return "";
  }
}

int UsockAddress::GetSockAddrLen() {
  switch (gen_addr_.ss_family) {
    case AF_INET:
      return sizeof(struct sockaddr_in);
    case AF_INET6:
      return sizeof(struct sockaddr_in6);
    case AF_LOCAL:
      return sizeof(struct sockaddr_un);
    default:
      LOG_SYSFATAL << "UsockAddress::GetSockAddrLen";
      return 0;
  }
}

uint32_t UsockAddress::IpNetEndian() const {
  assert(family() == AF_INET);
  return addr_.sin_addr.s_addr;
}

uint16_t UsockAddress::ToPort() const { return ntohs(PortNetEndian()); }

static __thread char t_resolveBuffer[64 * 1024];

bool UsockAddress::Resolve(const std::string& hostname, UsockAddress* out) {
  assert(out != NULL);
  struct hostent hent;
  struct hostent* he = NULL;
  int herrno = 0;
  bzero(&hent, sizeof(hent));

  int ret = gethostbyname_r(hostname.c_str(), &hent, t_resolveBuffer,
                            sizeof t_resolveBuffer, &he, &herrno);
  if (ret == 0 && he != NULL) {
    assert(he->h_addrtype == AF_INET && he->h_length == sizeof(uint32_t));
    out->addr_.sin_addr = *reinterpret_cast<struct in_addr*>(he->h_addr);
    return true;
  } else {
    if (ret) {
      LOG_SYSERR << "InetAddress::resolve";
    }
    return false;
  }
}

void UsockAddress::FromIpPort(const char* ip, uint16_t port,
                              struct sockaddr_in* addr) {
  addr->sin_family = AF_INET;
  addr->sin_port = htons(port);
  if (::inet_pton(AF_INET, ip, &addr->sin_addr) <= 0) {
    LOG_SYSERR << "sockets::fromIpPort";
  }
}

void UsockAddress::FromIpPort(const char* ip, uint16_t port,
                              struct sockaddr_in6* addr) {
  addr->sin6_family = AF_INET6;
  addr->sin6_port = htons(port);
  if (::inet_pton(AF_INET6, ip, &addr->sin6_addr) <= 0) {
    LOG_SYSERR << "sockets::fromIpPort";
  }
}

void UsockAddress::FromUnPath(const char* path, struct sockaddr_un* addr) {
  addr->sun_family = AF_LOCAL;
  strncpy(addr->sun_path, path, sizeof addr->sun_path);
}

void UsockAddress::ToIpPortUtil(char* buf, size_t size,
                                const struct sockaddr* addr) {
  ToIpUtil(buf, size, addr);
  size_t end = ::strlen(buf);
  const struct sockaddr_in* addr4 = sockaddr_in_cast(addr);
  uint16_t port = ntohs(addr4->sin_port);
  assert(size > end);
  snprintf(buf + end, size - end, ":%u", port);
}

void UsockAddress::ToIpUtil(char* buf, size_t size,
                            const struct sockaddr* addr) {
  if (addr->sa_family == AF_INET) {
    assert(size >= INET_ADDRSTRLEN);
    const struct sockaddr_in* addr4 = sockaddr_in_cast(addr);
    ::inet_ntop(AF_INET, &addr4->sin_addr, buf, static_cast<socklen_t>(size));
  } else if (addr->sa_family == AF_INET6) {
    assert(size >= INET6_ADDRSTRLEN);
    const struct sockaddr_in6* addr6 = sockaddr_in6_cast(addr);
    ::inet_ntop(AF_INET6, &addr6->sin6_addr, buf, static_cast<socklen_t>(size));
  }
}

}  // namespace uevent
