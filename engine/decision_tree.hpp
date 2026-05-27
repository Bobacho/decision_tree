#ifndef decision_tree_h
#define decision_tree_h

#include "../data/binary_tree.hpp"
#include "../data/dataset.hpp"
#include "../graphics/graphics.hpp"
#include <vector>

#include "impurity_type.hpp"

struct NodeInfo {
  bool is_leaf;
  int predicted_class;
  int split_feature;
  double split_threshold;
};

struct SplitInfo {
  int feature;
  double threshold;
  double gain;
};

class DecisionTree {
  using Indices = std::vector<int>;

  std::shared_ptr<BinaryTree<NodeInfo>> binaryTree;

  std::unique_ptr<Dataset> dataset;
  size_t max_depth = 1;

  ImpurityType impurity_type = GINI;

  void iterate();
  void build_tree(size_t current_depth,
                  std::shared_ptr<BinaryTree<NodeInfo>> curr, Indices indices);
  float calculate_threshold(size_t feature);
  int predict_one(const std::vector<float> &sample,
                  std::shared_ptr<BinaryTree<NodeInfo>> node);

  double calculate_impurity(const std::vector<int> &labels);

  SplitInfo find_best_split(const Values &X, const std::vector<int> &y,
                            const std::vector<int> indices);

public:
  DecisionTree(size_t max_depth);
  DecisionTree(size_t max_depth, ImpurityType type);
  void train(const Dataset &dataset);
  std::vector<int> predict(const Dataset &values);
  void print_tree();
  void draw(Graphics &g) const;
};

#endif
