/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <fstream>
#include <vector>

#if defined(__clang__) || defined(__GNUC__)
#define FASTTEXT_DEPRECATED(msg) __attribute__((__deprecated__(msg)))
#elif defined(_MSC_VER)
#define FASTTEXT_DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define FASTTEXT_DEPRECATED(msg)
#endif

namespace fasttext {

namespace utils {

int64_t size(std::ifstream&);

void seek(std::ifstream&, int64_t);

template <typename T>
bool contains(const std::vector<T>& container, const T& value) {
  return std::find(container.begin(), container.end(), value) !=
      container.end();
}

class ifstreams {
 protected:
  std::vector<std::ifstream> ss_;
  std::vector<int64_t> sizes_;
  int64_t size_;
  int32_t curr_;
 public:
  explicit ifstreams(const std::vector<std::string>&);
  ifstreams(const ifstreams&) = default;
  ifstreams& operator=(const ifstreams&) = delete;

  int64_t size() const;
  int32_t numFiles() const;

  void seek(int32_t, int32_t);

  std::ifstream& get();

  void close();
};


} // namespace utils

} // namespace fasttext
