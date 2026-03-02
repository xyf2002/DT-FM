#pragma once

#include <functional>
#include <iostream>
#include <list>
#include <stdexcept>
#include <unordered_map>
#include <utility>

template <typename KeyType, typename ValueType>
class FixCapLRUCache {
 public:
  explicit FixCapLRUCache(size_t cap) : capacity_(cap) {}

  void Put(const KeyType& key, const ValueType& value) {
    auto it = cache_.find(key);
    if (it != cache_.end()) {
      // Update item if it exists and move it to the back of the list
      lru_.erase(it->second.second);
    } else {
      // Check capacity_ and remove the least recently used item
      if (cache_.size() == capacity_) {
        cache_.erase(lru_.front());
        lru_.pop_front();
      }
    }
    // Insert new item at the back of the list
    lru_.push_back(key);
    cache_[key] = {value, --lru_.end()};
  }

  ValueType Get(const KeyType& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      throw std::range_error("Key not found");
    }
    // Move the accessed item to the back of the list
    lru_.erase(it->second.second);
    lru_.push_back(key);
    it->second.second = --lru_.end();
    return it->second.first;
  }

  bool Exist(const KeyType& key) const {
    return cache_.find(key) != cache_.end();
  }

 private:
  std::list<KeyType> lru_;  // Stores keys of cache items
  std::unordered_map<
      KeyType, std::pair<ValueType, typename std::list<KeyType>::iterator>>
      cache_;
  size_t capacity_;
};

template <typename KeyType, typename ValueType>
class FixSizeLRUCache {
 public:
  using DeleterType = std::function<void(const KeyType&, const ValueType&)>;

  explicit FixSizeLRUCache(size_t max_bytes, DeleterType deleter = nullptr)
      : max_bytes_(max_bytes), current_bytes_(0), deleter_(deleter) {}

  void Put(const KeyType& key, const ValueType& value, size_t value_size) {
    auto it = cache_.find(key);
    size_t item_size = sizeof(KeyType) + value_size;  // Calculate total size

    if (it != cache_.end()) {
      // If item exists, update it and adjust size
      current_bytes_ -= sizeof(KeyType) + it->second.first.second;
      lru_.erase(it->second.second);
      if (deleter_)
        deleter_(it->first,
                 it->second.first.first);  // Call deleter on old value
      cache_.erase(it);
    }

    // Evict least recently used items until there's enough space
    while (current_bytes_ + item_size > max_bytes_ && !lru_.empty()) {
      auto old_key = lru_.front();
      auto& old_value = cache_[old_key];
      current_bytes_ -= sizeof(KeyType) + old_value.first.second;
      if (deleter_)
        deleter_(old_key,
                 old_value.first.first);  // Call deleter on evicted value
      cache_.erase(old_key);
      lru_.pop_front();
    }

    // Insert new item at the back of the list
    lru_.push_back(key);
    cache_[key] = {{value, value_size}, --lru_.end()};
    current_bytes_ += item_size;
  }

  ValueType Get(const KeyType& key) {
    auto it = cache_.find(key);
    if (it == cache_.end()) {
      throw std::range_error("Key not found");
    }
    // Move the accessed item to the back of the list
    lru_.erase(it->second.second);
    lru_.push_back(key);
    it->second.second = --lru_.end();
    return it->second.first.first;
  }

  bool Exist(const KeyType& key) const {
    return cache_.find(key) != cache_.end();
  }

 private:
  std::list<KeyType> lru_;  // Stores keys of cache items
  // Map from key to a pair of (value, value size) and iterator to the position
  // in LRU list
  std::unordered_map<KeyType, std::pair<std::pair<ValueType, size_t>,
                                        typename std::list<KeyType>::iterator>>
      cache_;
  size_t max_bytes_;      // Maximum bytes allowed in the cache
  size_t current_bytes_;  // Current bytes used in the cache
  DeleterType deleter_;   // Function to call on item deletion
};
