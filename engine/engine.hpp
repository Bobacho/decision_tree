#ifndef engine_h
#define engine_h

#include "../data/dataset.hpp"

template <typename T> class Engine {
public:
  virtual void train(const Dataset &dataset) = 0;
  virtual T predict(const Values &values) = 0;
};
#endif
