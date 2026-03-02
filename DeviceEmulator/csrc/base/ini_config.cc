#include "ini_config.h"

#include <assert.h>
#include <glob.h>
#include <regex.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

namespace base {

using namespace std;

IniConfig::IniConfig() {}

IniConfig::~IniConfig() {}

void IniConfig::Clear() { m_mapContents.clear(); }

int IniConfig::LoadFromFile(const char* pFilename,
                            set<string>* pExcludePath /* = NULL*/) {
  FILE* pFile = fopen(pFilename, "r");
  if (NULL == pFile) {
    return -1;
  }
  // 只有首次调用的时候才更新m_strConfigFile，避免递归调用时更新错误
  if (pExcludePath == NULL) {
    m_strConfigFile = pFilename;
  }

  vector<string> vecLines;
  char szLineBuffer[2048];
  while (!feof(pFile) && !ferror(pFile)) {
    if (NULL != fgets(szLineBuffer, sizeof(szLineBuffer), pFile)) {
      vecLines.push_back(szLineBuffer);
    }
  }
  fclose(pFile);
  return LoadFromMemory(vecLines, pExcludePath);
}

void IniConfig::InsertItem(const string& strSection, const string& strKey,
                           const string& strValue) {
  multimap<string, multimap<string, string> >::iterator it =
      m_mapContents.find(strSection);
  if (m_mapContents.end() == it) {
    multimap<string, string> mapEmpty;
    it = m_mapContents.insert(make_pair(strSection, mapEmpty));
  }
  it->second.insert(make_pair(strKey, strValue));
}

int IniConfig::LoadFromMemory(const string& strContent,
                              set<string>* pExcludePath /* = NULL*/) {
  vector<string> vecLines;
  uint32_t iOffset = 0;
  const char* pTempStr = NULL;
  while ((pTempStr = strchr(strContent.c_str() + iOffset, '\n'))) {
    uint32_t iLength = pTempStr - (strContent.c_str() + iOffset);
    vecLines.push_back(string(strContent.c_str() + iOffset, iLength));
    iOffset += iLength + 1;
  }
  vecLines.push_back(strContent.c_str() + iOffset);
  return LoadFromMemory(vecLines, pExcludePath);
}

int IniConfig::LoadFromMemory(const vector<string>& vecLines,
                              set<string>* pExcludePath /* = NULL*/) {
  static regex_t sComment, sSection, sKeyValue, sKey;
  static const char* pCommentRegular = "(^[ \t]*([;#]|//))";
  static const char* pSectionRegular = "^[ \t]*[[](.+)[]][ \t]*";
  static const char* pKeyValueRegular =
      "^[ \t]*([^ \t=]+)[ \t=]+([^ \t=\r\n]+)";
  static const char* pKeyRegular = "^[ \t]*([^ \t=]+)";
  static bool bCompFlag = false;
  if (!bCompFlag) {
    if (0 != regcomp(&sComment, pCommentRegular, REG_EXTENDED | REG_NOSUB)) {
      return -1;
    }
    if (0 != regcomp(&sSection, pSectionRegular, REG_EXTENDED)) {
      regfree(&sComment);
      return -1;
    }
    if (0 != regcomp(&sKeyValue, pKeyValueRegular, REG_EXTENDED)) {
      regfree(&sComment);
      regfree(&sSection);
      return -1;
    }
    if (0 != regcomp(&sKey, pKeyRegular, REG_EXTENDED)) {
      regfree(&sComment);
      regfree(&sSection);
      regfree(&sKeyValue);
      return -1;
    }
    bCompFlag = true;
  }

  regmatch_t matches[64];
  string strCurrentSection;
  string strKey, strValue;
  for (unsigned i = 0; i < vecLines.size(); i++) {
    const char* pContent = vecLines[i].c_str();
    // 如果匹配了注释行则忽略
    if (0 == regexec(&sComment, pContent, sizeof(matches) / sizeof(regmatch_t),
                     matches, 0)) {
      continue;
    }
    // 如果匹配了段，则更新当前段
    if (0 == regexec(&sSection, pContent, sizeof(matches) / sizeof(regmatch_t),
                     matches, 0)) {
      assert(matches[1].rm_so != -1 && matches[1].rm_eo != -1);
      strCurrentSection.assign(pContent + matches[1].rm_so,
                               matches[1].rm_eo - matches[1].rm_so);
      continue;
    }
    // 如果匹配了KeyValue则插入
    if (0 == regexec(&sKeyValue, pContent, sizeof(matches) / sizeof(regmatch_t),
                     matches, 0)) {
      assert(matches[2].rm_so != -1 && matches[2].rm_eo != -1);
      strKey.assign(pContent + matches[1].rm_so,
                    matches[1].rm_eo - matches[1].rm_so);
      strValue.assign(pContent + matches[2].rm_so,
                      matches[2].rm_eo - matches[2].rm_so);
      InsertItem(strCurrentSection, strKey, strValue);
      continue;
    }
    // 如果匹配了Key则插入
    if (0 == regexec(&sKey, pContent, sizeof(matches) / sizeof(regmatch_t),
                     matches, 0)) {
      assert(matches[1].rm_so != -1 && matches[1].rm_eo != -1);
      strKey.assign(pContent + matches[1].rm_so,
                    matches[1].rm_eo - matches[1].rm_so);
      InsertItem(strCurrentSection, strKey, "");
      continue;
    }
    // 忽略
    continue;
  }

  set<string> setExcludePath;
  if (!pExcludePath) {
    pExcludePath = &setExcludePath;
  }

  for (unsigned i = 0; i < Count("", "include"); i++) {
    // 查找Include
    string strInclude = GetValue("", "include", i);
    // 如果已经处理过这个include了则忽略
    if (pExcludePath->find(strInclude) != pExcludePath->end()) {
      continue;
    }

    // 将处理过的路径增加到排除路径列表中，避免重复处理
    pExcludePath->insert(strInclude);

    // 处理该路径下的每个符合条件的文件
    glob_t glob_result;
    glob(strInclude.c_str(), GLOB_TILDE, NULL, &glob_result);
    for (unsigned int i = 0; i < glob_result.gl_pathc; i++) {
      if (0 != LoadFromFile(glob_result.gl_pathv[i], pExcludePath)) {
        return -1;
      }
    }
    globfree(&glob_result);
  }
  return 0;
}

const string IniConfig::TranslateValue(const string& strValue) {
  string __return;
  __return = strValue;

  // 翻译环境变量
  static const char* pRegular = "([$@])[{(]([^)}]*)[})]";
  static regex_t sReg;
  static bool comp_flag = false;
  if (!comp_flag) {
    if (0 != regcomp(&sReg, pRegular, REG_EXTENDED | REG_ICASE)) {
      return __return;
    }
    comp_flag = true;
  }

  regmatch_t szMatch[20];
  while (1) {
    int iRet = regexec(&sReg, __return.c_str(),
                       sizeof(szMatch) / sizeof(regmatch_t), szMatch, 0);
    if (iRet != 0) {
      return __return;
    }
    string strFirstPart, strMiddlePart, strLastPart;
    if (szMatch[0].rm_so == 0) {
      strFirstPart = "";
    } else {
      strFirstPart =
          string(__return.c_str(), __return.c_str() + szMatch[0].rm_so);
    }
    string strValType = string(__return.c_str() + szMatch[1].rm_so,
                               __return.c_str() + szMatch[1].rm_eo);
    string strPreChange = string(__return.c_str() + szMatch[2].rm_so,
                                 __return.c_str() + szMatch[2].rm_eo);
    if (strValType.compare("$") == 0) {
      strMiddlePart = getenv(strPreChange.c_str());
    } else if (strValType.compare("@") == 0) {
      strMiddlePart = GetValueByFullName(strPreChange.c_str());
    } else {
      strMiddlePart = strPreChange;
    }
    strLastPart = string(__return.c_str() + szMatch[0].rm_eo);

    __return = strFirstPart + strMiddlePart + strLastPart;
  }
  return __return;
}

uint32_t IniConfig::Count(const string& strSection, const string& strKey) {
  multimap<string, multimap<string, string> >::iterator it =
      m_mapContents.find(strSection);
  if (it == m_mapContents.end()) {
    return 0;
  }
  return it->second.count(strKey);
}

int64_t IniConfig::IntValue(const string& strSection, const string& strKey,
                            int iIndex) {
  string strValue = GetValue(strSection, strKey, iIndex);
  int64_t iReturn;
  const char* pValue = strValue.c_str();
  char* pEndPtr = NULL;
  if (strValue.size() > 2 && pValue[0] == '0' &&
      (pValue[1] == 'x' || pValue[1] == 'X')) {
    iReturn = strtoll(pValue + 2, &pEndPtr, 16);
  } else {
    iReturn = strtoll(pValue, &pEndPtr, 10);
  }
  if (pEndPtr) {
    // 忽略空格和tab
    while (pEndPtr[0] == ' ' || pEndPtr[0] == '\t') {
      pEndPtr++;
    }
    switch (pEndPtr[0]) {
      case 'k':
      case 'K':
        iReturn *= 1024;
        break;
      case 'm':
      case 'M':
        iReturn *= 1024 * 1024;
        break;
      case 'g':
      case 'G':
        iReturn *= 1024l * 1024l * 1024l;
        break;
      case 't':
      case 'T':
        iReturn *= 1024l * 1024l * 1024l * 1024l;
        break;
    }
  }
  return iReturn;
}

const string IniConfig::GetValueByFullName(const string& strKey, int iIndex) {
  const char* pDot = strchr(strKey.c_str(), '.');
  if (NULL == pDot) {
    return GetValue("", strKey, iIndex);
  } else {
    string strSection;
    strSection.assign(strKey.c_str(), pDot - strKey.c_str());
    return GetValue(strSection, pDot + 1, iIndex);
  }
}

const string IniConfig::GetValue(const string& strSection, const string& strKey,
                                 int iIndex) {
  string __temp_default;
  multimap<string, multimap<string, string> >::iterator it =
      m_mapContents.find(strSection);
  if (it == m_mapContents.end()) {
    return __temp_default;
  }
  pair<multimap<string, string>::iterator, multimap<string, string>::iterator>
      ret = it->second.equal_range(strKey);
  multimap<string, string>::iterator it2 = ret.first;
  int iCounter = 0;
  for (; it2 != ret.second; it2++) {
    if (iCounter++ == iIndex) {
      return TranslateValue(it2->second);
    }
  }
  return __temp_default;
}

int IniConfig::Reload() {
  if (m_strConfigFile.size() == 0) {
    return 0;
  }
  IniConfig sOther;
  if (0 != sOther.LoadFromFile(m_strConfigFile.c_str())) {
    return -1;
  }
  *this = sOther;
  return 0;
}

}  // namespace base
