#include "graphics.hpp"
#include <GL/glut.h>
#include <cmath>
#include <mutex>
#include <string>
#include <unistd.h>

// -------------------------------------------------------------------
// Global instance for the GLUT callback (corrected to avoid recreation)
static std::shared_ptr<Graphics> g_current_instance =
    std::make_shared<Graphics>();

Graphics::Graphics() { this->request_queue = std::queue<Request>(); }

// Display callback – now simply calls loop() on the existing instance
static void display_wrapper() {
  if (g_current_instance) {
    g_current_instance->render();
  }
}

// -------------------------------------------------------------------
// Drawing functions (thread-safe: only modify request_ and post redisplay)

void Graphics::draw_circle(const Position &pos, unsigned int radius,
                           const Position &center) {
  Request request_;
  request_.pos = pos;
  request_.radius = static_cast<double>(radius);
  request_.center = center;
  request_.type = GraphicType::CIRCLE;
  g_current_instance->request_queue.push(request_);
}

void Graphics::draw_rectangle(const Position &pos, unsigned int width,
                              unsigned int height) {
  Request request_;
  request_.pos = pos;
  request_.width = static_cast<double>(width);
  request_.height = static_cast<double>(height);
  request_.type = GraphicType::RECTANGLE;
  g_current_instance->request_queue.push(request_);
}

void Graphics::draw_line(const Position &pos, const Position &dest) {
  Request request_;
  request_.pos = pos;
  request_.dest = dest;
  request_.type = GraphicType::LINE;
  g_current_instance->request_queue.push(request_);
}

void Graphics::render_string(const Position &pos, const std::string &message) {
  Request request_;
  request_.pos = pos;
  request_.string = message;
  request_.type = GraphicType::STRING;
  g_current_instance->request_queue.push(request_);
}

// -------------------------------------------------------------------
// OpenGL rendering (called only from the GLUT main thread)
void Graphics::render() {
  glClear(GL_COLOR_BUFFER_BIT);
  glColor3f(1.0f, 1.0f, 1.0f);

  while (!g_current_instance->request_queue.empty()) {
    Request request_ = g_current_instance->request_queue.front();
    g_current_instance->request_queue.pop();
    switch (request_.get_type()) {
    case GraphicType::LINE:
      if (request_.dest.has_value()) {
        glBegin(GL_LINES);
        glVertex2i(request_.pos.first, request_.pos.second);
        glVertex2i(request_.dest->first, request_.dest->second);
        glEnd();
      }
      break;

    case GraphicType::CIRCLE:
      if (request_.radius.has_value() && request_.center.has_value()) {
        const double r = request_.radius.value();
        const auto &c = request_.center.value();
        glBegin(GL_LINE_LOOP);
        for (double angle = 0.0; angle < 2 * M_PI; angle += 0.1) {
          glVertex2i(c.first + static_cast<int>(r * std::cos(angle)),
                     c.second + static_cast<int>(r * std::sin(angle)));
        }
        glEnd();
      }
      break;

    case GraphicType::RECTANGLE:
      if (request_.width.has_value() && request_.height.has_value()) {
        const int x = request_.pos.first;
        const int y = request_.pos.second;
        const int w = static_cast<int>(request_.width.value());
        const int h = static_cast<int>(request_.height.value());
        glBegin(GL_LINE_LOOP);
        glVertex2i(x, y);
        glVertex2i(x + w, y);
        glVertex2i(x + w, y + h);
        glVertex2i(x, y + h);
        glEnd();
      }
      break;

    case GraphicType::STRING:
      if (request_.string.has_value()) {
        glRasterPos2i(request_.pos.first, request_.pos.second);
        for (char c : request_.string.value()) {
          glutBitmapCharacter(GLUT_BITMAP_HELVETICA_12, c);
        }
      }
      break;

    default:
      break;
    }
  }
  glutSwapBuffers();
}

// -------------------------------------------------------------------
// GLUT initialisation (runs in its own thread)
void Graphics::show() {
  if (initialized)
    return;
  initialized = true;

  // GLUT initialisation – must happen in the same thread that calls show()
  int argc = 0;
  glutInit(&argc, nullptr);
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowSize(800, 600);
  glutCreateWindow("Decision Tree");
  glutDisplayFunc(display_wrapper);

  // Set an orthographic projection so that pixel coordinates work
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0, 800, 0, 600, -1, 1);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();
  glutMainLoop(); // blocks until window is closed
}
