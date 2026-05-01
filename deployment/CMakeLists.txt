cmake_minimum_required(VERSION 3.5)

set(PROJECT_NAME Yolov12Deployment)
project(${PROJECT_NAME} VERSION 0.0.1 LANGUAGES CXX)

# -------------- Support C++17 ------------------#
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# -------------- OpenCV  ------------------#
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# -------------- CUDA ------------------#
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()

# -------------- ONNXRUNTIME  ------------------#
# Use GPU-enabled ONNX Runtime (for CUDAExecutionProvider)
set(ONNXRUNTIME_VERSION 1.22.0)
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}")

if (NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNX Runtime not found at ${ONNXRUNTIME_ROOT}. Please run download_ort.sh")
endif()

include_directories(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/include)

# -------------- Sources ------------------#
set(PROJECT_SOURCES
    main.cpp
    inference.cpp
    inference.h
    sei_parser.cpp
    sei_parser.h
    offline_processor.cpp
    offline_processor.h
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})
add_executable(${PROJECT_NAME}CurrentBaseline
    main_current_baseline.cpp
    inference.cpp
    inference.h
    sei_parser.cpp
    sei_parser.h
    offline_processor_current_baseline.cpp
    offline_processor_current_baseline.h
)
add_executable(${PROJECT_NAME}Duo
    main_duo.cpp
    inference.cpp
    inference.h
    sei_parser.cpp
    sei_parser.h
    offline_processor.cpp
    offline_processor.h
    offline_processor_duo.cpp
    offline_processor_duo.h
)

# -------------- SSL ------------------#
find_package(OpenSSL REQUIRED)
include_directories(${OPENSSL_INCLUDE_DIR})

# -------------- Linking ------------------#
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${PROJECT_NAME}CurrentBaseline ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so OpenSSL::SSL OpenSSL::Crypto)
target_link_libraries(${PROJECT_NAME}Duo ${OpenCV_LIBS} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so OpenSSL::SSL OpenSSL::Crypto)
if (USE_CUDA)
    # CUDA provider is a plugin .so loaded by ORT, but linking it ensures ldd/packaging catches it.
    target_link_libraries(${PROJECT_NAME} ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so)
    target_link_libraries(${PROJECT_NAME}CurrentBaseline ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so)
    target_link_libraries(${PROJECT_NAME}Duo ${ONNXRUNTIME_ROOT}/lib/libonnxruntime_providers_cuda.so)
endif ()
if (USE_CUDA)
    target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
    target_link_libraries(${PROJECT_NAME}CurrentBaseline ${CUDA_LIBRARIES})
    target_link_libraries(${PROJECT_NAME}Duo ${CUDA_LIBRARIES})
endif ()

# -------------- Resources ------------------#
# Download COCO yaml
# configure_file(../examples/YOLOv8-ONNXRuntime-CPP/coco.yaml ${CMAKE_CURRENT_BINARY_DIR}/coco.yaml COPYONLY)

# Copy the trained model (assuming it's in the standard location)
# If the file doesn't exist at config time, this might warn/fail, but we can copy manually too.
# We'll expect the user to provide the model path via CLI, but we can copy a default if present.
