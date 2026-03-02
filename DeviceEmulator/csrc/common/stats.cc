#include "stats.h"

#include <iomanip>
#include <limits>
#include <sstream>

using namespace std;

/**************************** HistogramBuckets ********************************/
HistogramBuckets::HistogramBuckets(const string& name, int64_t bucket_size,
                                   int64_t min, int64_t max)
    : name_(name),
      bucket_size_(bucket_size),
      min_(min),
      max_(max),
      min_value_(numeric_limits<int64_t>::max()),
      max_value_(numeric_limits<int64_t>::min()),
      count_(0) {
  auto bucket_count = (max - min) / bucket_size;
  if (bucket_count * bucket_size < max - min) ++bucket_count;
  // for below min and above max
  bucket_count += 2;
  buckets_.assign(bucket_count, Bucket());
}

void HistogramBuckets::AddValue(int64_t value) {
  lock(mutex_, lk_percent_);
  // make sure both already-locked mutexes are unlocked at the end of scope
  lock_guard<mutex> lock1(mutex_, adopt_lock);
  lock_guard<mutex> lock2(lk_percent_, adopt_lock);

  auto& bucket = GetByValue(value);
  bucket.count += 1;
  bucket.sum += value;
  if (value < min_value_) min_value_ = value;
  if (value > max_value_) max_value_ = value;
  count_ += 1;
}

void HistogramBuckets::AddValue(int64_t value, uint64_t count) {
  lock(mutex_, lk_percent_);
  // make sure both already-locked mutexes are unlocked at the end of scope
  lock_guard<mutex> lock1(mutex_, adopt_lock);
  lock_guard<mutex> lock2(lk_percent_, adopt_lock);

  auto& bucket = GetByValue(value);
  bucket.count += count;
  bucket.sum += value * count;
  if (value < min_value_) min_value_ = value;
  if (value > max_value_) max_value_ = value;
  count_ += count;
}

void HistogramBuckets::Clear() {
  lock_guard<mutex> g(mutex_);
  for (auto& bucket : buckets_) {
    bucket.Clear();
  }
  min_value_ = numeric_limits<int64_t>::max();
  max_value_ = numeric_limits<int64_t>::min();
  count_ = 0;
}

string HistogramBuckets::OutputString() const {
  lock_guard<mutex> g(mutex_);
  ostringstream oss;

  oss << left << setw(36) << name_ << left << setw(24) << count_ << left
      << setw(24) << min_value_ << left << setw(24) << max_value_ << left
      << setw(24) << GetPercentileEstimate(0.5) << left << setw(24)
      << GetPercentileEstimate(0.9) << left << setw(24)
      << GetPercentileEstimate(0.95) << left << setw(24)
      << GetPercentileEstimate(0.99) << endl;

  if (count_ == 0) {
    return "";
  }
  return oss.str();
}

int64_t HistogramBuckets::GetPercentileEstimate(double pct) const {
  lock_guard<mutex> g(lk_percent_);

  double low_pct = 0.0;
  double high_pct = 0.0;
  size_t idx = GetPercentileBucketIdx(pct, &low_pct, &high_pct);
  if (low_pct == 0.0 && high_pct == 0.0) {
    // means all buckets are empty
    return 0;
  }
  if (low_pct == high_pct) {
    return buckets_[idx].Avg();
  }

  int64_t avg = buckets_[idx].Avg();
  int64_t low;
  int64_t high;
  if (idx == 0) {
    if (avg > min_) {
      // Unlikely to happen except overflow happen
      return GetBucketMin(idx);
    }
    high = min_;
    low = high - (2 * (high - avg));
    if (low > avg) low = numeric_limits<int64_t>::min();
  } else if (idx == buckets_.size() - 1) {
    if (avg < max_) {
      // Unlikely to happen except overflow happen
      return GetBucketMax(idx);
    }
    low = max_;
    high = low + (2 * (avg - low));
    if (high < avg) high = numeric_limits<int64_t>::max();
  } else {
    low = GetBucketMin(idx);
    high = GetBucketMax(idx);
    if (avg < low || avg > high) {
      // Unlikely to happen except overflow happen
      return (low + high) / 2;
    }
  }

  double median_pct = (low_pct + high_pct) / 2;
  if (pct < median_pct) {
    double pct_through_section = (pct - low_pct) / (median_pct - low_pct);
    return low + (avg - low) * pct_through_section;
  } else {
    double pct_through_section = (pct - median_pct) / (high_pct - median_pct);
    return avg + (high - avg) * pct_through_section;
  }
}

size_t HistogramBuckets::GetPercentileBucketIdx(double pct, double* low_pct,
                                                double* high_pct) const {
  auto bucket_count = buckets_.size();
  vector<uint64_t> counts(bucket_count);
  uint64_t total_count = 0;
  for (size_t n = 0; n < bucket_count; ++n) {
    uint64_t count = buckets_[n].count;
    counts[n] = count;
    total_count += count;
  }

  if (total_count == 0) {
    *low_pct = 0.0;
    *high_pct = 0.0;
    return 1;
  }

  double prev_pct = 0.0;
  double cur_pct = 0.0;
  uint64_t cur_count = 0;
  size_t idx = 0;
  for (idx = 0; idx < bucket_count; ++idx) {
    if (counts[idx] == 0) continue;
    prev_pct = cur_pct;
    cur_count += counts[idx];
    cur_pct = static_cast<double>(cur_count) / total_count;
    if (pct < cur_pct) break;
  }
  *low_pct = prev_pct;
  *high_pct = cur_pct;
  return idx;
}

size_t HistogramBuckets::GetBucketIdx(int64_t value) const {
  if (value < min_) {
    return 0;
  } else if (value >= max_) {
    return buckets_.size() - 1;
  } else {
    return (value - min_) / bucket_size_ + 1;
  }
}

Bucket& HistogramBuckets::GetByValue(int64_t value) {
  return buckets_[GetBucketIdx(value)];
}

const Bucket& HistogramBuckets::GetByValue(int64_t value) const {
  return buckets_[GetBucketIdx(value)];
}

int64_t HistogramBuckets::GetBucketMin(size_t idx) const {
  if (idx == 0) {
    return numeric_limits<int64_t>::min();
  }
  if (idx == buckets_.size() - 1) {
    return max_;
  }
  return (min_ + (idx - 1) * bucket_size_);
}

int64_t HistogramBuckets::GetBucketMax(size_t idx) const {
  if (idx == buckets_.size() - 1) {
    return numeric_limits<int64_t>::max();
  }
  return (min_ + idx * bucket_size_);
}

/******************************** Histogram ***********************************/

ServiceStats* ServiceStats::self_ = new ServiceStats();

ServiceStats* ServiceStats::Instance() { return self_; }

ServiceStats::ServiceStats() {}

ServiceStats::~ServiceStats() {}

int ServiceStats::Initialize() {
#define ADD_HISTOGRAM(name, step, min, max) \
  histograms_.push_back(make_shared<HistogramBuckets>(#name, step, min, max))

  ADD_HISTOGRAM(SRV_INTERNAL_ERR, 50, 0, 1000000);

  // Performance
  ADD_HISTOGRAM(SRV_TOTAL_QUERY, 1, 0, 1000000);
  ADD_HISTOGRAM(SRV_TOTAL_TRAFFIC, 1, 0, 1000000);

#undef ADD_HISTOGRAM

  return 0;
}

void ServiceStats::Stat(ServiceStatType type, int64_t value) {
  histograms_[type]->AddValue(value);
}

void ServiceStats::Stat(ServiceStatType type, int64_t value, uint64_t count) {
  histograms_[type]->AddValue(value, count);
}

string ServiceStats::OutputString() const {
  ostringstream oss;
  oss << left << setw(36) << "op" << left << setw(24) << "count" << left
      << setw(24) << "min" << left << setw(24) << "max" << left << setw(24)
      << ".5" << left << setw(24) << ".9" << left << setw(24) << ".95" << left
      << setw(24) << ".99" << endl;
  for (auto& histogram : histograms_) {
    oss << histogram->OutputString();
  }
  return oss.str();
}

void ServiceStats::Clear() {
  for (auto& histogram : histograms_) {
    histogram->Clear();
  }
}
