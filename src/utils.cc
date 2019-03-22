/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "utils.h"

#include <ios>

namespace fasttext {

namespace utils {

int64_t size(std::ifstream& ifs) {
  ifs.seekg(std::streamoff(0), std::ios::end);
  return ifs.tellg();
}

void seek(std::ifstream& ifs, int64_t pos) {
  ifs.clear();
  ifs.seekg(std::streampos(pos));
}

ifstreams::ifstreams(const std::vector<std::string>& files) {
  if (files.empty()) {
    throw std::invalid_argument("no files for training!");
  }
  size_ = 0;
  for (auto fn : files) {
    std::ifstream fis0(fn);
    ss_.emplace_back(fn);
    auto& fis = ss_[ss_.size() - 1];
    if (!fis.is_open()) {
      throw std::invalid_argument(
          fn + " cannot be opened for training!");
    }
    int64_t thisSize = utils::size(fis);
    sizes_.push_back(thisSize);
    size_ += thisSize;
  }
  curr_ = 0;
}

int64_t ifstreams::size() const {
  return size_;
}

void ifstreams::seek(int32_t i, int32_t n) {
  auto pos = (int64_t)(((double)size_ / n) * i);
  int j = 0;
  for (; pos > 0; j++) {
    pos -= sizes_[j];
  }
  if (pos >= 0) {
    auto& ifs = ss_[j];
    utils::seek(ifs, pos);
  } else {
    auto& ifs = ss_[j - 1];
    pos = sizes_[j - 1] + pos;
    utils::seek(ifs, pos);
  }
}

std::ifstream& ifstreams::get() {
  if (ss_[curr_].eof()) {
    utils::seek(ss_[curr_], 0);
    curr_ = static_cast<int32_t>((curr_ + 1) % ss_.size());
    utils::seek(ss_[curr_], 0);
  }
  return ss_[curr_];
}

void ifstreams::close() {
  for (auto& s : ss_) {
    s.close();
  }
}

int32_t ifstreams::numFiles() const {
  return ss_.size();
}
} // namespace utils

} // namespace fasttext
