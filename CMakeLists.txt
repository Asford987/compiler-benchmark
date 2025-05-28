cmake_minimum_required(VERSION 3.15)
project(NNBenchmark CXX)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_COMPILEDNN "Build with CompiledNN backend" ON)
option(BUILD_TENSORFLOW "Build with TensorFlow C++ backend" OFF)
option(BUILD_TVM        "Build with Apache TVM backend" OFF)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/external)

# CompiledNN
if(BUILD_COMPILEDNN)
    add_subdirectory(external/CompiledNN)
    add_executable(compilednn_runner src/main_compilednn.cpp)
    target_link_libraries(compilednn_runner CompiledNN)
    find_package(OpenCV REQUIRED)
    target_link_libraries(compilednn_runner ${OpenCV_LIBS})

endif()

# TensorFlow
if(BUILD_TENSORFLOW)
    include_directories(external/tensorflow/include)
    link_directories(external/tensorflow/lib)

    add_executable(tensorflow_runner src/main_tensorflow.cpp)
    target_link_libraries(tensorflow_runner
        tensorflow_cc
        tensorflow_framework
        pthread
        dl
    )
endif()

# Apache TVM
if(BUILD_TVM)
    include_directories(external/tvm/include)
    include_directories(external/tvm/3rdparty/dmlc-core/include)
    include_directories(external/tvm/3rdparty/tvm/src/runtime)
    link_directories(external/tvm/build)

    add_executable(tvm_runner src/main_tvm.cpp)
    target_link_libraries(tvm_runner
        tvm_runtime
        dl
        pthread
    )
endif()
