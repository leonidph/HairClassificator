cmake_minimum_required(VERSION 3.15)
project(HairClassificator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(mlpack REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgcodecs imgproc)

add_executable(hair-classificator src/main.cpp)
target_link_libraries(hair-classificator PRIVATE mlpack armadillo ${OpenCV_LIBS})
