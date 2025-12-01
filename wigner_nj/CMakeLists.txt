cmake_minimum_required(VERSION 3.28)

project(wigner_nj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# eigen
include_directories(${CMAKE_SOURCE_DIR}/external/eigen)
# fmt
add_subdirectory(${CMAKE_SOURCE_DIR}/external/fmt)
#WignerSymbol
include_directories(${CMAKE_SOURCE_DIR}/external/WignerSymbol)
# head files
# include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(wigner_nj main.cpp)
target_link_libraries(wigner_nj PRIVATE fmt::fmt)