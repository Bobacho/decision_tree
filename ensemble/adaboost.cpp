#include "adaboost.hpp"
#include <cmath>
#include <numeric>
#include <random>

// ---------------------------------------------------------------------
// AdaBoost implementation
// ---------------------------------------------------------------------
AdaBoost::AdaBoost(int n_estimators, size_t tree_max_depth)
    : n_estimators_(n_estimators), tree_max_depth_(tree_max_depth) {}

void AdaBoost::train(const Dataset &dataset) {
  const auto &values = dataset.get_values();
  const size_t N = values->size();
  if (N == 0)
    return;

  // Extract true labels (assumed to be the last column)
  std::vector<int> y(N);
  for (size_t i = 0; i < N; ++i) {
    y[i] = static_cast<int>(values->at(i).back());
  }

  // Initialize sample weights uniformly
  std::vector<double> weights(N, 1.0 / N);

  trees_.clear();
  alphas_.clear();
  trees_.reserve(n_estimators_);
  alphas_.reserve(n_estimators_);

  std::mt19937 rng(std::random_device{}());

  for (int t = 0; t < n_estimators_; ++t) {
    // --- Step 1: Create bootstrap dataset according to current weights ---
    Dataset sampled_dataset =
        Dataset(); // labelTarget = 1 (unused, but required by constructor)
    std::discrete_distribution<int> dist(weights.begin(), weights.end());
    for (size_t i = 0; i < N; ++i) {
      int idx = dist(rng);
      sampled_dataset.addValues(values->at(idx));
    }
    // Set training_set to all indices (0..N-1) because the sampled dataset
    // contains exactly N rows (some may be repeated)
    sampled_dataset.training_set.resize(N);
    std::iota(sampled_dataset.training_set.begin(),
              sampled_dataset.training_set.end(), 0);

    // --- Step 2: Train a weak learner on this bootstrap sample ---
    auto tree = std::make_shared<DecisionTree>(tree_max_depth_);
    tree->train(sampled_dataset);

    // --- Step 3: Compute weighted error ---
    std::vector<int> predictions = tree->predict(sampled_dataset);
    double weighted_error = 0.0;
    for (size_t i = 0; i < N; ++i) {
      // map 0/1 to -1/1 for error calculation
      int true_label = (y[i] == 0 ? -1 : 1);
      int pred_label = (predictions[i] == 0 ? -1 : 1);
      if (true_label != pred_label) {
        weighted_error += weights[i];
      }
    }
    // Normalize error (weights sum to 1)
    weighted_error /= std::accumulate(weights.begin(), weights.end(), 0.0);

    // Avoid division by zero or negative gain
    if (weighted_error <= 0.0) {
      // perfect classifier – stop early
      alphas_.push_back(1e10);
      trees_.push_back(tree);
      break;
    }
    double alpha = 0.5 * std::log((1.0 - weighted_error) / weighted_error);
    alphas_.push_back(alpha);
    trees_.push_back(tree);

    // --- Step 5: Update sample weights ---
    double total_weight = 0.0;
    for (size_t i = 0; i < N; ++i) {
      int true_label = (y[i] == 0 ? -1 : 1);
      int pred_label = (predictions[i] == 0 ? -1 : 1);
      if (true_label != pred_label) {
        weights[i] *= std::exp(alpha);
      } else {
        weights[i] *= std::exp(-alpha);
      }
      total_weight += weights[i];
    }
    // Normalise
    for (auto &w : weights)
      w /= total_weight;
  }
}

std::vector<int> AdaBoost::predict(const Dataset &dataset) const {
  const auto &values = dataset.get_values();
  size_t N = values->size();
  if (N == 0 || trees_.empty())
    return {};

  // For each sample, accumulate weighted votes
  std::vector<double> scores(N, 0.0);
  for (size_t t = 0; t < trees_.size(); ++t) {
    std::vector<int> pred = trees_[t]->predict(dataset);
    double alpha = alphas_[t];
    for (size_t i = 0; i < N; ++i) {
      // convert 0/1 to -1/1 vote
      int vote = (pred[i] == 0 ? -1 : 1);
      scores[i] += alpha * vote;
    }
  }

  // Final class: sign of the weighted sum
  std::vector<int> result(N);
  for (size_t i = 0; i < N; ++i) {
    result[i] = (scores[i] >= 0.0) ? 1 : 0;
  }
  return result;
}
