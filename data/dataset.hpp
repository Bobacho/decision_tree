#ifndef dataset_hpp
#define dataset_hpp

#include <memory>
#include <optional>
#include <string>
#include <vector>

using Values = std::vector<std::vector<float>>;
using Labels = std::vector<std::string>;

class Dataset {
  Labels labels;
  Values values;

  size_t labelTarget;
  void load(const std::string &path);

  Dataset(const size_t &labelTarget);

public:
  Dataset(const size_t &labelTarget, const std::string &path);
  void buildTest(const float &testLikelihood);

  size_t size() const;
  std::vector<int> test_set;
  std::vector<int> training_set;

  const std::unique_ptr<Values>
  head(const std::optional<size_t> &startOpt) const;
  const std::unique_ptr<Values> head() const;

  const std::unique_ptr<Dataset> subset() const;

  const std::unique_ptr<Labels> get_labels() const;
  const std::unique_ptr<Values> get_values() const;
  const size_t classTarget() const;

  void addValues(std::vector<float> row);
};

#endif
