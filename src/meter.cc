/**
 * Copyright (c) 2016-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "meter.h"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

namespace fasttext {

void Meter::log(
    const std::vector<int32_t>& labels,
    const std::vector<std::pair<real, int32_t>>& predictions,
    loss_name loss) {
  nexamples_++;

  if (loss == loss_name::sigmoid) {
    labelScores_[labels[0]].emplace_back(predictions[labels[0]].first, labels[1]);
    labelMetrics_[labels[0]].predicted++;
    if (labels[1] == 1) {
      labelMetrics_[labels[0]].gold++;
    }
  } else {
    metrics_.gold += labels.size();
    metrics_.predicted += predictions.size();

    for (const auto& prediction : predictions) {
      labelMetrics_[prediction.second].predicted++;

      if (utils::contains(labels, prediction.second)) {
        labelMetrics_[prediction.second].predictedGold++;
        metrics_.predictedGold++;
      }
    }

    for (const auto& label : labels) {
      labelMetrics_[label].gold++;
    }
  }
}

double Meter::precision(int32_t i) {
  return labelMetrics_[i].precision();
}

double Meter::recall(int32_t i) {
  return labelMetrics_[i].recall();
}

double Meter::f1Score(int32_t i) {
  return labelMetrics_[i].f1Score();
}

int64_t Meter::negatives(int32_t i) {
  return labelMetrics_[i].predicted - labelMetrics_[i].gold;
}
int64_t Meter::positives(int32_t i) {
  return labelMetrics_[i].gold;
}

double Meter::auc(int32_t li) {
  auto& scores = labelScores_[li];
  std::sort(scores.begin(), scores.end());
  std::vector<real> rank(scores.size());
  for (int64_t i = 0; i < scores.size(); i++) {
    if (i == scores.size() - 1 || scores[i].first != scores[i + 1].first) {
      rank[i] = i + 1;
    } else {
      int64_t j = i + 1;
      while (j < scores.size() && scores[j].first == scores[i].first) {
        j += 1;
      }
      real r = (i + 1 + j) / 2.0f;
      for (int64_t k = i; k < j; k++) {
        rank[k] = r;
      }
      i = j - 1;
    }
  }
  double auc = 0;
  for (int64_t i = 0; i < scores.size(); i++) {
    if (scores[i].second == 1) {
      auc += rank[i];
    }
  }
  int64_t pos = positives(li);
  int64_t neg = negatives(li);
  return (auc - (pos * (pos + 1) / 2.0)) / (pos * neg);
}

double Meter::precision() const {
  return metrics_.precision();
}

double Meter::recall() const {
  return metrics_.recall();
}

void Meter::writeGeneralMetrics(std::ostream& out, int32_t k) const {
  out << "N"
      << "\t" << nexamples_ << std::endl;
  out << std::setprecision(3);
  out << "P@" << k << "\t" << metrics_.precision() << std::endl;
  out << "R@" << k << "\t" << metrics_.recall() << std::endl;
}

} // namespace fasttext
