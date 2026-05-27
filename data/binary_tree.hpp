#ifndef binary_tree_h
#define binary_tree_h

#include <memory>

template <typename T> class BinaryTree {
public:
  BinaryTree() : left(nullptr), right(nullptr) {}
  // or just: BinaryTree() = default;

  std::shared_ptr<BinaryTree> left;
  std::shared_ptr<BinaryTree> right;
  T value;
};

#endif
