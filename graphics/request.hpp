#ifndef request_hpp
#define request_hpp

#include "graphic_type.hpp"
#include <array>
#include <memory>
#include <optional>
#include <string>

using Vertex = std::array<double, 2>;

class Request {

public:
  std::pair<int, int> pos;
  /*STRING REQUEST */
  std::optional<std::string> string = std::nullopt;
  /*CIRCLE REQUEST */
  std::optional<double> radius = std::nullopt;
  std::optional<std::pair<int, int>> center = std::nullopt;
  /*RECTANGLE REQUEST */
  std::optional<double> width = std::nullopt;
  std::optional<double> height = std::nullopt;
  /*LINE REQUEST */
  std::optional<std::pair<int, int>> dest = std::nullopt;
  Request();
  GraphicType type;

  const GraphicType get_type() const;
};

#endif
