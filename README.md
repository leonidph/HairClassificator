# HairClassificator

C++ project using CMake, mlpack, and OpenCV.

## Build

```bash
mkdir -p build
cd build
cmake ..
cmake --build .
```

## Run

```bash
./hair-classificator
```

This project trains a simple logistic regression model with mlpack and writes a sample OpenCV image to `output.png`.

> Requires `mlpack`, `Armadillo`, and `OpenCV` installed on your system.
