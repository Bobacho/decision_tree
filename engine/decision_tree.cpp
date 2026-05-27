#include "decision_tree.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>

DecisionTree::DecisionTree(size_t max_depth) {
  this->max_depth = max_depth;
  this->binaryTree = std::make_unique<BinaryTree<NodeInfo>>();
}
DecisionTree::DecisionTree(size_t max_depth, ImpurityType type) {
  this->max_depth = max_depth;
  this->impurity_type = type;
  this->binaryTree = std::make_unique<BinaryTree<NodeInfo>>();
}

void DecisionTree::train(const Dataset &dataset) {
  size_t size = dataset.size();
  auto labels = dataset.get_labels();
  auto values = dataset.get_values();

  this->dataset = std::make_unique<Dataset>(dataset);

  std::vector<int> all_indices = {};
  all_indices.reserve(dataset.training_set.size());
  for (size_t i = 0; i < dataset.training_set.size(); ++i) {
    all_indices.push_back(i);
  }

  build_tree(0, binaryTree, all_indices);
}

// La clase retorna la siguiente estructura [
// 	<Label, Conteo>, Label Mayoritaria
// ]
std::pair<std::unordered_map<int, int>, int>
class_count(const std::vector<int> &labels) {
  std::unordered_map<int, int> counts;
  for (int label : labels) {
    counts[label]++;
  }
  int majority = -1;
  int max_count = -1;
  for (const auto &pair : counts) {
    if (pair.second > max_count) {
      max_count = pair.second;
      majority = pair.first;
    }
  }
  return {counts, majority};
}

double gini_impurity(const std::vector<int> &labels) {
  auto [counts, _] = class_count(labels);
  double impurity = 1.0;
  double n = labels.size();
  for (const auto &pair : counts) {
    double p = pair.second / n;
    impurity -= p * p;
  }
  return impurity;
}

double entropy_impurity(const std::vector<int> &labels) {
  auto [counts, _] = class_count(labels);
  double n = labels.size();
  double impurity = 0.0;
  for (const auto &pair : counts) {
    double p = pair.second / n;
    impurity += p * std::log(p);
  }
  return -impurity;
}

double DecisionTree::calculate_impurity(const std::vector<int> &labels) {
  switch (this->impurity_type) {
  case GINI:
    return gini_impurity(labels);
  case ENTROPY:
    return entropy_impurity(labels);
  default:
    return -1;
  }
}

SplitInfo DecisionTree::find_best_split(const Values &X,
                                        const std::vector<int> &y,
                                        const std::vector<int> indices) {

  double impurity = calculate_impurity(y);
  SplitInfo best = {};
  best.feature = -1;
  best.threshold = 0.0;
  best.gain = 0.0;

  size_t n_features = X.at(0).size() - 1;
  for (size_t f = 0; f < n_features; ++f) {
    std::vector<float> values;
    values.reserve(indices.size());
    for (int idx : indices) {
      values.push_back(X.at(idx)[f]);
    }
    size_t n_total = indices.size();
    std::sort(values.begin(), values.end());
    for (size_t i = 0; i < values.size() - 1; ++i) {
      double threshold = (values[i] + values[i + 1]) / 2.0;
      std::vector<int> left_idx = {};
      left_idx.reserve(indices.size() / 2);
      std::vector<int> right_idx = {};
      right_idx.reserve(indices.size() / 2);
      for (int idx : indices) {
        if (X.at(idx)[f] <= threshold) {
          left_idx.push_back(idx);
        } else {
          right_idx.push_back(idx);
        }
      }
      if (left_idx.empty() || right_idx.empty()) {
        continue;
      }
      std::vector<int> left_labels = {};
      std::vector<int> right_labels = {};
      left_labels.reserve(left_idx.size());
      right_labels.reserve(right_idx.size());
      for (int idx : left_idx) {
        left_labels.push_back(X.at(idx).at(X.at(idx).size() - 1));
      }
      for (int idx : right_idx) {
        right_labels.push_back(X.at(idx).at(X.at(idx).size() - 1));
      }
      double left_imp = gini_impurity(left_labels);
      double right_imp = gini_impurity(right_labels);
      double w = (double)left_idx.size() / n_total;
      double gain = impurity - (w * left_imp + (1.0 - w) * right_imp);
      if (gain > best.gain) {
        best.gain = gain;
        best.feature = f;
        best.threshold = threshold;
      }
    }
  }
  return best;
}

void print_node(NodeInfo node) {
  std::cout << "-------------" << "\n";
  std::cout << "Is leaf:" << node.is_leaf << "\n";
  std::cout << "Threshold:" << node.split_threshold << "\n";
  std::cout << "Feature:" << node.split_feature << "\n";
  std::cout << "Predicted class:" << node.predicted_class << "\n";
  std::cout << "-------------" << "\n";
}

void recurse(BinaryTree<NodeInfo> &binaryTree) {
  print_node(binaryTree.value);
  std::cout << "Node: " << &binaryTree << "\n";
  if (binaryTree.left != nullptr) {
    std::cout << "Left of node: " << &binaryTree << "\n";
    recurse(*binaryTree.left);
  }
  if (binaryTree.right != nullptr) {
    std::cout << "Right of node: " << &binaryTree << "\n";
    recurse(*binaryTree.right);
  }
}

void DecisionTree::print_tree() {
  BinaryTree<NodeInfo> binaryTree = *this->binaryTree;
  recurse(binaryTree);
}

const std::unique_ptr<Values> get_training_values(const Dataset &dataset) {
  const auto &values = dataset.get_values();
  Values result = {};
  result.reserve(dataset.training_set.size());
  for (auto &trainingIdx : dataset.training_set) {
    result.push_back(values->at(trainingIdx));
  }
  return std::make_unique<Values>(result);
}

const std::vector<int> get_labels_values(const Dataset &dataset) {
  const auto &values = dataset.get_values();
  std::vector<int> result = {};
  result.reserve(dataset.training_set.size());
  for (auto &trainingIdx : dataset.training_set) {
    result.push_back(values->at(trainingIdx).at(values->at(0).size() - 1));
  }
  return result;
}

void DecisionTree::build_tree(size_t current_depth,
                              std::shared_ptr<BinaryTree<NodeInfo>> curr,
                              Indices indices) {

  const auto &x = get_training_values(*dataset);
  const auto &labels = get_labels_values(*dataset);

  auto [counts, majority] = class_count(labels);

  if (current_depth > max_depth) {
    curr->value.is_leaf = true;
    curr->value.predicted_class = majority;
    return;
  }

  if (counts.size() == 1) {
    curr->value.is_leaf = true;
    curr->value.predicted_class = majority;
    return;
  }
  if (indices.size() < 2) {
    curr->value.is_leaf = true;
    curr->value.predicted_class = majority;
    return;
  }

  SplitInfo split = find_best_split(*x, labels, indices);
  // If no beneficial split found, make leaf
  if (split.gain <= 0.0 || split.feature == -1) {
    curr->value.is_leaf = true;
    curr->value.predicted_class = majority;
    return;
  }

  // --- Internal node: set split rule ---
  curr->value.is_leaf = false;
  curr->value.split_feature = split.feature;
  curr->value.split_threshold = split.threshold;

  // Partition indices into left/right based on the best split
  std::vector<int> left_indices, right_indices;
  for (size_t idx : indices) {
    if (x->at(idx)[split.feature] <= split.threshold)
      left_indices.push_back(idx);
    else
      right_indices.push_back(idx);
  }

  // Create child nodes
  curr->left = std::make_shared<BinaryTree<NodeInfo>>();
  curr->right = std::make_shared<BinaryTree<NodeInfo>>();
  build_tree(current_depth + 1, curr->left, left_indices);
  build_tree(current_depth + 1, curr->right, right_indices);
}

int DecisionTree::predict_one(const std::vector<float> &sample,
                              std::shared_ptr<BinaryTree<NodeInfo>> node) {
  if (node->value.is_leaf)
    return node->value.predicted_class;
  if (sample[node->value.split_feature] <= node->value.split_threshold)
    return predict_one(sample, node->left);
  else
    return predict_one(sample, node->right);
}

std::vector<int> DecisionTree::predict(const Dataset &data) {
  std::vector<int> preds;
  const auto &X = data.get_values();
  for (size_t i = 0; i < X->size(); ++i) {
    preds.push_back(predict_one(X->at(i), this->binaryTree));
  }
  return preds;
}
// Helper: recursively draw a node and its children
static void draw_node(Graphics &g,
                      const std::shared_ptr<BinaryTree<NodeInfo>> &node, int x,
                      int y, int x_offset) {
  const int BOX_W = 80, BOX_H = 30;

  // Draw the node rectangle
  g.draw_rectangle({x - BOX_W / 2, y - BOX_H / 2}, BOX_W, BOX_H);

  // Build the text to display inside the node
  std::string text;
  if (node->value.is_leaf) {
    text = "Class " + std::to_string(node->value.predicted_class);
  } else {
    text = "F" + std::to_string(node->value.split_feature) +
           " <= " + std::to_string(node->value.split_threshold).substr(0, 4);
  }

  // Draw the text (centred roughly inside the box)
  // Position slightly offset to avoid overlapping the box border
  g.render_string({x - static_cast<int>(text.size() * 3), y - 6}, text);

  // If internal node, draw lines to children and recurse
  if (!node->value.is_leaf && node->left && node->right) {
    int child_y = y - 100;
    int left_x = x - x_offset;
    int right_x = x + x_offset;

    g.draw_line({x, y - BOX_H / 2}, {left_x, child_y + BOX_H / 2});
    g.draw_line({x, y - BOX_H / 2}, {right_x, child_y + BOX_H / 2});

    draw_node(g, node->left, left_x, child_y, x_offset / 2);
    draw_node(g, node->right, right_x, child_y, x_offset / 2);
  }
}

void DecisionTree::draw(Graphics &g) const {
  // Start at root, top centre of the window (assumes ~800x600)
  draw_node(g, binaryTree, 400, 550, 200);
}
