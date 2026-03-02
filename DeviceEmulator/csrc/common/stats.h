#pragma once

#include <stdint.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/noncopyable.h"

struct Bucket {
  Bucket() : sum(0), count(0) {}
  void Clear() {
    sum = 0;
    count = 0;
  }
  void Add(int64_t s, uint64_t c) {
    sum += s;
    count += c;
  }
  int64_t Avg() const {
    if (count == 0) return 0;
    return static_cast<double>(sum) / static_cast<double>(count);
  }
  int64_t sum;
  uint64_t count;
};

class HistogramBuckets {
 public:
  HistogramBuckets(const std::string& name, int64_t bucket_size, int64_t min,
                   int64_t max);
  void AddValue(int64_t value);
  void AddValue(int64_t value, uint64_t count);
  void Clear();
  std::string OutputString() const;

 public:
  std::string GetName() { return name_; }
  uint64_t GetCount() {
    std::lock_guard<std::mutex> g(mutex_);
    return count_;
  }
  Bucket& GetByValue(int64_t value);
  const Bucket& GetByValue(int64_t value) const;
  std::size_t GetBucketIdx(int64_t value) const;
  int64_t GetPercentileEstimate(double pct) const;
  std::size_t GetPercentileBucketIdx(double pct, double* low_pct,
                                     double* high_pct) const;
  int64_t GetBucketMin(std::size_t idx) const;
  int64_t GetBucketMax(std::size_t idx) const;

 private:
  std::string name_;
  uint64_t bucket_size_;
  int64_t min_;
  int64_t max_;
  int64_t min_value_;
  int64_t max_value_;
  uint64_t count_;
  std::vector<Bucket> buckets_;
  mutable std::mutex mutex_;
  mutable std::mutex lk_percent_;
};

typedef std::shared_ptr<HistogramBuckets> HistogramBucketsPtr;

enum ServiceStatType : size_t {
  // ERR
  SRV_INTERNAL_ERR = 0,

  // Performance
  SRV_TOTAL_QUERY,
  SRV_TOTAL_TRAFFIC,

  SRV_STAT_TYPE_FINISH
};

class ServiceStats : public base::noncopyable {
  ServiceStats();
  ~ServiceStats();

 public:
  static ServiceStats* Instance();
  int Initialize();
  void Stat(ServiceStatType type, int64_t value);
  void Stat(ServiceStatType type, int64_t value, uint64_t count);
  std::string OutputString() const;
  void Clear();

  uint64_t GetCount(ServiceStatType type) const {
    return histograms_[type]->GetCount();
  }
  std::string GetName(ServiceStatType type) const {
    return histograms_[type]->GetName();
  }
  int64_t GetPercentile90(ServiceStatType type) const {
    return histograms_[type]->GetPercentileEstimate(0.9);
  }

 private:
  std::vector<HistogramBucketsPtr> histograms_;
  static ServiceStats* self_;
};

// #define VAL_NAME(val) #val
#define SRV_STATS ServiceStats::Instance()
#define RECORD_SRV_STATS(type, val) SRV_STATS->Stat(type, val)
#define RECORD_SRV_COUNT(type, count) SRV_STATS->Stat(type, 0, count)
#define RECORD_OP_LATENCY(type)                                            \
  {                                                                        \
    auto now = chrono::system_clock::now();                                \
    auto dop =                                                             \
        chrono::duration_cast<chrono::microseconds>(now - session_delay_); \
    RECORD_SRV_STATS(type, dop.count());                                   \
  }
#define RECORD_STEP(type)                       \
  {                                             \
    auto dop = DURATION_UNTIL_NOW(m_prev_time); \
    RECORD_ACCESS_STAT(type, dop.count());      \
    m_prev_time = TIME_NOW;                     \
  }
