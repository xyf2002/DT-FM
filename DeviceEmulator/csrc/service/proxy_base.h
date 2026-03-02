#pragma once

#include <chrono>
#include <functional>
#include <map>

#include "network/uevent.h"

class ProxyBase : public std::enable_shared_from_this<ProxyBase> {
 public:
  static const int kServiceContinue = 0;
  static const int kServiceFinished = 1;

 protected:
  static const double kStepTimeout;

 public:
  ProxyBase();
  virtual ~ProxyBase();

 public:
  void set_in_conn(const uevent::ConnectionUeventPtr& conn) { in_conn_ = conn; }
  void set_header(const ProxyHeaderPtr& header) { header_ = header; }

 public:
  virtual void Process(const RpcMsgPtr&) = 0;

 protected:
  void SetError(int retcode, const std::string& err_msg) {
    retcode_ = retcode;
    err_msg_ = err_msg;
  }

 protected:
  RpcMsgPtr rpc_msg_;
  uint32_t ctx_id_;
  int retcode_;
  uevent::ConnectionUeventPtr in_conn_;
  std::string err_msg_;
  std::string session_;
  ProxyHeaderPtr header_;
};

typedef std::function<ProxyBase*()> CreateProxyIns;

#define DEFAULT_SERVICE_MEMBER(ClassName)                        \
  ClassName();                                                   \
  virtual ~ClassName();                                          \
  virtual const char* ServiceName() const { return #ClassName; } \
  static ProxyBase* CreateMyself() { return new ClassName(); }   \
  virtual void Process(const RpcMsgPtr& rpc_msg);

#define LOG_SERVICE_DEBUG                                           \
  LOG_DEBUG << "machine:" << ServiceName() << "(" << ctx_id_ << ")" \
            << "|session:" << session_ << "|"

#define LOG_SERVICE_INFO                                           \
  LOG_INFO << "machine:" << ServiceName() << "(" << ctx_id_ << ")" \
           << "|session:" << session_ << "|"

#define LOG_SERVICE_ERROR                                           \
  LOG_ERROR << "machine:" << ServiceName() << "(" << ctx_id_ << ")" \
            << "|session:" << session_ << "|"
