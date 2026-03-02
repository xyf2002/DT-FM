#ifndef __MY_UUID_H_
#define __MY_UUID_H_

#include <uuid/uuid.h>

#include <string>

namespace base {

class MyUuid {
 public:
  MyUuid();
  MyUuid(uuid_t sUuid);
  MyUuid(const char* pString, unsigned iSize);

  MyUuid(const MyUuid& sOther);
  MyUuid& operator=(const MyUuid& sOther);
  MyUuid& operator=(const uuid_t& sOther);

  bool operator<(const MyUuid& sCmp) const;

  const uuid_t* const Uuid() const { return &m_szUuid; }

  const char* UuidString() const { return m_szUuidStr; }

  static std::string NewUuid();

 protected:
  uuid_t m_szUuid;
  char m_szUuidStr[37];
};

}  // namespace base
#endif
