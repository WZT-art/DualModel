cmake_minimum_required(VERSION 3.20)

project(yolo11trt LANGUAGES CXX CUDA)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

set(TENSORRT_PATH "D:/TensorRT10.10-cu124")
set(OpenCV_DIR "D:/OpenCV410/install/x64/vc17/lib")
set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4")
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_DIR})
set(cudnn_DIR "D:/cudnn9.2-cu12")
# set(Boost_DIR "D:/boost_1_86_0")
set(Boost_ROOT "D:/boost_1_82_0")
# set(Python3_ROOT_DIR "E:/Miniconda/envs/yolo11")
# set(Qt6_DIR "E:/Qt6.8.2/6.8.3/msvc2022_64/lib/cmake/Qt6")
set(PCL_DIR "D:/PCL 1.14.1/cmake")
# set(pybind11_DIR "E:/Miniconda/envs/yolo11/Lib/site-packages/pybind11/share/cmake/pybind11")

set(Boost_USE_STATIC_LIBS OFF)  
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_LIBRARY_DIR "${Boost_ROOT}/lib64-msvc-14.3")

add_definitions(-DBOOST_ALL_NO_LIB -DBOOST_AUTO_LINK_NOMANGLE)

# set(CMAKE_AUTOMOC ON)
# set(CMAKE_AUTOUIC ON)
# set(CMAKE_AUTORCC ON)

# set(BOOST_INCLUDE ${Boost_DIR})
# set(BOOST_LIBRARY_DIR "${Boost_DIR}/lib64-msvc-14.3")

# Windows 特定配置
# if(MSVC)
#     set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
#     add_definitions(
#         -D_WIN32_WINNT=0x0601  # Windows 7 or later
#         -DWIN32_LEAN_AND_MEAN
#     )
# endif()

# find_package(CUDA EXACT 12.4 REQUIRED)
find_package(Boost EXACT 1.82.0 REQUIRED COMPONENTS log
												    log_setup
												    thread
												    system
												    filesystem
												    date_time)
find_package(PCL EXACT 1.14.1 REQUIRED)

find_package(CUDAToolkit REQUIRED)

# find_package(Qt6 EXACT 6.8.3 REQUIRED COMPONENTS Core Widgets)
# find_package(Python3 COMPONENTS Interpreter Development REQUIRED)


if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
find_package(OpenCV EXACT 4.10.0 REQUIRED)

# find_package(pybind11 REQUIRED)

set(SOURCES

)

add_executable(${PROJECT_NAME} ${SOURCES} main.cpp)
# set_target_properties(${PROJECT_NAME} PROPERTIES
# 	AUTOMOC ON
# 	AUTOUIC ON
# 	AUTORCC ON
# )

# target_compile_definitions(${PROJECT_NAME} PUBLIC API_EXPORTS)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}/include
    ${cudnn_DIR}/include
	${Boost_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
	# ${Python3_INCLUDE_DIRS}
)

file(GLOB TENSORRT_LIBS "${TENSORRT_PATH}/lib/*.lib")
file(GLOB cudnn_LIBS "${cudnn_DIR}/lib/x64/*.lib")
# message(STATUS "Found TensorRT libs: ${TENSORRT_LIBS}")
# message(STATUS "Found cuDNN libs: ${cudnn_LIBS}")

# set(BOOST_LIBS
# 	${BOOST_LIBRARY_DIR}/boost_log-vc143-mt-x64-1_86.lib
# 	${BOOST_LIBRARY_DIR}/boost_log_setup-vc143-mt-x64-1_86.lib
# 	${BOOST_LIBRARY_DIR}/boost_thread-vc143-mt-x64-1_86.lib
# 	${BOOST_LIBRARY_DIR}/boost_system-vc143-mt-x64-1_86.lib
# 	${BOOST_LIBRARY_DIR}/boost_filesystem-vc143-mt-x64-1_86.lib
# 	${BOOST_LIBRARY_DIR}/boost_date_time-vc143-mt-x64-1_86.lib
# )

target_link_libraries(${PROJECT_NAME} PUBLIC
	${OpenCV_LIBS}
	${CUDA_LIBRARIES}
	CUDA::cudart
	CUDA::cublas
	${TENSORRT_LIBS}
    ${cudnn_LIBS}
	# ${BOOST_LIBS}
	Boost::log
	Boost::log_setup
	Boost::thread
	Boost::system
	Boost::filesystem
	Boost::date_time
	${PCL_LIBRARIES}
)

# message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# message(STATUS "Boost libraries: ${Boost_LIBRARIES}")
# message(STATUS "PCL libraries: ${PCL_LIBRARIES}")
# message(STATUS "PCL include directories: ${PCL_INCLUDE_DIRS}")
# get_target_property(flann_inc_dirs FLANN::FLANN INTERFACE_INCLUDE_DIRECTORIES)
# message(STATUS "FLANN include dirs from target: ${flann_inc_dirs}")
# message(STATUS "pybind11 include directories: ${pybind11_INCLUDE_DIRS}")
# message(STATUS "Python3 include directories: ${Python3_INCLUDE_DIRS}")
# message(STATUS "Python3 libraries: ${Python3_LIBRARIES}")
# message(STATUS "Qt6 found: ${Qt6_FOUND}")
# message(STATUS "Qt6 version: ${Qt6_VERSION}")
