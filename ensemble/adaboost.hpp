#ifndef ADABOOST_H
#define ADABOOST_H

#include "../data/dataset.hpp"
#include "../engine/decision_tree.hpp"
#include <memory>
#include <vector>

class AdaBoost {
public:
  // Constructor: number of weak learners (iterations)
  explicit AdaBoost(int n_estimators, size_t tree_max_depth = 2);

  // Train the ensemble on a dataset
  void train(const Dataset &dataset);

  // Predict class labels (0/1) for a dataset
  std::vector<int> predict(const Dataset &dataset) const;

  // Access the trained trees and their alphas (for debugging/visualization)
  const std::vector<std::shared_ptr<DecisionTree>> &get_trees() const {
    return trees_;
  }
  const std::vector<double> &get_alphas() const { return alphas_; }

private:
  int n_estimators_;      // number of boosting rounds
  size_t tree_max_depth_; // max depth for each weak tree
  std::vector<std::shared_ptr<DecisionTree>> trees_; // ensemble
  std::vector<double> alphas_;                       // weight of each tree

  // Weighted sampling with replacement: returns a new Dataset containing
  // n_samples rows drawn according to the given probabilities.
  Dataset sample_with_weights(const Dataset &original,
                              const std::vector<double> &weights) const;
};

#endif // ADABOOST_H
