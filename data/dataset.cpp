#include "dataset.hpp"
#include <optional>
#ifndef random_h
#define random_h
#include <random>
#endif
#ifndef string_h
#define string_h
#include <filesystem>
#endif
#include <string>
#ifndef fstream_h
#define fstream_h
#include <fstream>
#endif

Dataset::Dataset(const size_t &labelTarget) { this->labelTarget = labelTarget; }

Dataset::Dataset() {}

Dataset::Dataset(const size_t &labelTarget, const std::string &path) {
  this->labelTarget = labelTarget;
  load(path);
  this->test_set.reserve(0.2 * this->values.size());
  this->training_set.reserve(0.8 * this->values.size());
}

std::vector<std::string> split(std::string v, char delimiter) {
  std::stringstream ss(v);
  std::string buffer;
  std::vector<std::string> result = {};
  result.reserve(v.size());
  while (std::getline(ss, buffer, delimiter)) {
    result.push_back(buffer);
  }
  return result;
}

void Dataset::load(const std::string &path) {
  const size_t base_lenght_raw_value = 5;
  std::ifstream in(path);
  size_t size = std::filesystem::file_size(path);
  if (in.fail()) {
    throw std::runtime_error("Dataset no pudo cargar los datos");
  }
  std::string line;
  std::getline(in, line);
  std::vector<std::string> labels = {};
  labels.reserve(line.size());
  labels = split(line, ',');
  if (labels.size() == 0) {
    throw std::runtime_error("Labels no se encontro labels");
  }
  this->values.reserve(size / (base_lenght_raw_value * labels.size()));
  this->labels = labels;
  while (std::getline(in, line)) {
    std::vector<std::string> raw_values = split(line, ',');
    std::vector<float> row = {};
    row.reserve(raw_values.size());
    for (auto &raw_value : raw_values) {
      row.push_back(std::stof(raw_value));
    }
    this->values.push_back(row);
  }
  in.close();
}

void Dataset::buildTest(const float &testLikelihood) {
  size_t size = this->values.size();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis{};
  for (int i = 0; i < size; ++i) {
    float value = dis(gen);
    if (value <= 0.8) {
      this->training_set.push_back(i);
    } else {
      this->test_set.push_back(i);
    }
  }
}

const std::unique_ptr<Values>
Dataset::head(const std::optional<size_t> &startOpt) const {
  Values values = {};
  values.reserve(5);
  size_t start = 0;
  if (startOpt.has_value()) {
    start = startOpt.value();
  }
  for (size_t i = start; i < start + 5; ++i) {
    values.push_back(this->values[i]);
  }
  return std::make_unique<Values>(values);
}
const std::unique_ptr<Values> Dataset::head() const {
  Values values = {};
  values.reserve(5);
  size_t start = 0;
  for (size_t i = start; i < start + 5; ++i) {
    values.push_back(this->values[i]);
  }
  return std::make_unique<Values>(values);
}

size_t Dataset::size() const { return this->values.size(); }

const std::unique_ptr<Labels> Dataset::get_labels() const {
  return std::make_unique<Labels>(this->labels);
}

const std::unique_ptr<Values> Dataset::get_values() const {
  return std::make_unique<Values>(this->values);
}
const size_t Dataset::classTarget() const { return this->labelTarget; }

const std::unique_ptr<Dataset> Dataset::subset() const {
  auto dataset = Dataset(1);
  Values values = {};
  values.reserve(this->test_set.size());
  int max_length = 5;
  int i = 0;
  for (auto &test : this->test_set) {
    if (i == max_length) {
      break;
    }
    values.push_back(this->get_values()->at(test));
    i++;
  }
  dataset.values = values;
  dataset.labels = this->labels;
  return std::make_unique<Dataset>(dataset);
}

void Dataset::addValues(std::vector<float> row) {
  this->values.push_back(std::move(row));
}
