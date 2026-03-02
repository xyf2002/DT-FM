#include "my_uuid.h"

#include <assert.h>
#include <string.h>

using namespace std;
using namespace base;

MyUuid::MyUuid() {
  uuid_generate(m_szUuid);
  uuid_unparse(m_szUuid, m_szUuidStr);
}

MyUuid::MyUuid(uuid_t szUuid) {
  uuid_copy(m_szUuid, szUuid);
  uuid_unparse(m_szUuid, m_szUuidStr);
}

MyUuid::MyUuid(const char* pString, unsigned iSize) {
  assert(iSize == 36);
  memcpy(m_szUuidStr, pString, 36);
  m_szUuidStr[36] = '\0';

  uuid_parse(m_szUuidStr, m_szUuid);
}

MyUuid::MyUuid(const MyUuid& sOther) {
  uuid_copy(m_szUuid, sOther.m_szUuid);
  strcpy(m_szUuidStr, sOther.m_szUuidStr);
}

MyUuid& MyUuid::operator=(const MyUuid& sOther) {
  if (this == &sOther) {
    return *this;
  }
  uuid_copy(m_szUuid, sOther.m_szUuid);
  strcpy(m_szUuidStr, sOther.m_szUuidStr);
  return *this;
}
MyUuid& MyUuid::operator=(const uuid_t& sOther) {
  uuid_copy(m_szUuid, sOther);
  uuid_unparse(m_szUuid, m_szUuidStr);
  return *this;
}

bool MyUuid::operator<(const MyUuid& sCmp) const {
  return uuid_compare(m_szUuid, sCmp.m_szUuid);
}

string MyUuid::NewUuid() {
  uuid_t sUuid;
  char szTemp[64];
  uuid_generate(sUuid);
  uuid_unparse(sUuid, szTemp);
  return szTemp;
}
