#ifndef INI_CONFIG_H
#define INI_CONFIG_H

#include <stdint.h>

#include <map>
#include <set>
#include <string>
#include <vector>

namespace base {

class IniConfig {
 public:
  IniConfig();
  ~IniConfig();

  void Clear();
  int LoadFromFile(const char* pFilename,
                   std::set<std::string>* pExcludePath = NULL);
  int LoadFromMemory(const std::vector<std::string>& vecLines,
                     std::set<std::string>* pExcludePath = NULL);

  int LoadFromMemory(const std::string& strContent,
                     std::set<std::string>* pExcludePath = NULL);

  int64_t IntValue(const std::string& strSection, const std::string& strKey,
                   int iIndex = 0);
  const std::string GetValue(const std::string& strSection,
                             const std::string& strKey, int iIndex = 0);
  const std::string GetValueByFullName(const std::string& strKey,
                                       int iIndex = 0);
  uint32_t Count(const std::string& strSection, const std::string& strKey);

  int Reload();

  const std::multimap<std::string, std::multimap<std::string, std::string> >&
  AllItems() const {
    return m_mapContents;
  }

  const std::string TranslateValue(const std::string& strValue);

  void InsertItem(const std::string& strSection, const std::string& strKey,
                  const std::string& strValue);

 private:
  // <section, <key, value> >
  std::multimap<std::string, std::multimap<std::string, std::string> >
      m_mapContents;
  std::string m_strConfigFile;
};

}  // namespace base
#endif
