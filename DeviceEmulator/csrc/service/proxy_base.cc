#include "service/proxy_base.h"

#include "base/my_uuid.h"
#include "common/generator.h"

using namespace std;
using namespace uevent;

ProxyBase::ProxyBase()
    : ctx_id_(NumGenerator::ctx_id()), retcode_(kSuccess), in_conn_(nullptr) {}

ProxyBase::~ProxyBase() {}
