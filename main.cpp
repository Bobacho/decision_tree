#include "data/dataset.hpp"
#include "engine/decision_tree.hpp"
#include "ensemble/adaboost.hpp"
#include "graphics/graphics.hpp"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#define fstream_h

void printValues(const Values &values);
template <typename T> void printVector(const std::vector<T> &val);
void dataset_summary(Dataset &dataset);

int main(int argc, char **argv) {
  int max_depth = 5;
  if (argc > 1) {
    try {
      max_depth = std::stoi(std::string(argv[1]));
    } catch (std::runtime_error r) {
      std::cout << "No inserto una profundidad valida\n";
    }
  }

  Graphics g = Graphics();

  std::unique_ptr<Dataset> dataset =
      std::make_unique<Dataset>(5, "./heart.csv");
  dataset->buildTest(0.8);

  dataset_summary(*dataset);

  std::unique_ptr<DecisionTree> decisiontree =
      std::make_unique<DecisionTree>(max_depth, ENTROPY);
  decisiontree->train(*dataset);

  decisiontree->print_tree();
  auto prediction = decisiontree->predict(*dataset->subset());
  printValues(*dataset->subset()->get_values());
  std::cout << "Prediccion del Decision Tree:\n";
  printVector(prediction);
  decisiontree->draw(g);
  auto adaboost = std::make_shared<AdaBoost>(5);
  adaboost->train(*dataset);

  prediction = adaboost->predict(*dataset->subset());
  std::cout << "Prediccion del Adaboost:\n";
  printVector(prediction);
  g.show();
}

void printValues(const Values &values) {
  for (auto &row : values) {
    std::cout << "[";
    for (size_t i = 0; i < row.size(); ++i) {
      std::cout << row[i];
      if (i < row.size() - 1) {
        std::cout << ","; // Only print comma if not the last element
      }
    }
    std::cout << "]\n"; // Or just use std::endl
  }
}

template <typename T> void printVector(const std::vector<T> &val) {
  std::cout << "[";
  for (size_t i = 0; i < val.size(); ++i) {
    std::cout << val[i];
    if (i != val.size() - 1) {
      std::cout << ",";
    }
  }
  std::cout << "]\n";
}

void dataset_summary(Dataset &dataset) {
  std::cout << "Size of dataset:" << dataset.size() << "\n";

  std::cout << "Labels:\n";
  printVector(*dataset.get_labels());
  std::cout << "Values:\n";
  printValues(*dataset.head());
}
