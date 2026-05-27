#ifndef graphics_hpp
#define graphics_hpp

#include <GL/freeglut_std.h>
#include <memory>

#include "graphic_type.hpp"
#include "request.hpp"
#include <cmath>
#include <queue>
#include <thread>
#include <utility>

using Position = std::pair<int, int>;
class Graphics {
  // … existing members (request_, request_mutex_, etc.) …
  std::queue<Request> request_queue;
  bool initialized = false;

public:
  Graphics(); // no thread anymore
  ~Graphics() = default;

  void render();

  void draw_line(const Position &pos, const Position &dest);
  void draw_circle(const Position &pos, unsigned int radius,
                   const Position &center);
  void draw_rectangle(const Position &pos, unsigned int width,
                      unsigned int height);
  void render_string(const Position &pos, const std::string &message);

  void show(); // blocks in glutMainLoop
               // … remove the old init() and thread_ …
};

#endif
