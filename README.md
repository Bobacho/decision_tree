# decision_tree

Este repositorio implementa un modelo de **Árbol de Decisión** en C++ para tareas de clasificación. Incluye:
- Un módulo para cargar y preprocesar datos (`data/`).
- La lógica principal para aprender y predecir con el árbol (`engine/`).
- Una visualización simple con OpenGL/GLUT que dibuja el árbol entrenado (`graphics/`).

El programa principal lee un conjunto de datos desde `data.csv`, entrena un árbol de decisión con profundidad configurable y criterio de impureza (índice de Gini o entropía), y luego muestra el árbol gráficamente.

---

## Dependencias

Antes de compilar, asegúrate de tener instaladas las siguientes librerías:

| Librería       | Paquete (ejemplo para Ubuntu/Debian) |
| :------------- | :----------------------------------- |
| **OpenGL**     | `libgl1-mesa-dev`                    |
| **GLUT**       | `freeglut3-dev`                      |
| **Librería estándar de C++** (filesystem, random, etc.) | – |

El código utiliza características de **C++17** (`std::filesystem`, `std::optional`), por lo que se necesita un compilador reciente (por ejemplo, GCC 9+ o Clang 10+).

---

## Compilación

Para compilar el proyecto, debes enlazar:
- La librería OpenGL (`-lGL`).
- La librería GLUT (`-lglut`).
- La librería matemática (`-lm`).
- Opcionalmente, la librería de filesystem de C++ (`-lstdc++fs`) en versiones antiguas de GCC.

### Usando g++

```bash
g++ -std=c++17 \
    main.cpp \
    data/dataset.cpp \
    engine/decision_tree.cpp \
    graphics/graphics.cpp \
    graphics/request.cpp \
    ensemble/adaboost.cpp \
    -o decision_tree \
    -lGL -lglut -lm
```

### Usando clang

```bash
clang++ -std=c++17 \
    main.cpp \
    data/dataset.cpp \
    engine/decision_tree.cpp \
    graphics/graphics.cpp \
    graphics/request.cpp \
    ensemble/adaboost.cpp \
    -o decision_tree \
    -lGL -lglut -lm
```
