#include "request.hpp"
#include "graphic_type.hpp"

Request::Request() {}

const GraphicType Request::get_type() const { return this->type; }
