cmake_minimum_required(VERSION 3.20)

set(CUDA_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8")
set(CUDA_TOOLKIT_ROOT_DIR ${CUDA_DIR})
project(yolo11trt LANGUAGES C CXX CUDA)

set(CMAKE_CUDA_STANDARD 17) 
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE Release)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build)

if(WIN32)
	link_libraries(ws2_32)
endif()

set(BOOST_ROOT "F:/boost_1_78_0")
# set(BOOST_ROOT "E:/boost1_84")

# Static Libraries
set(Boost_USE_STATIC_LIBS ON)  
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
set(Boost_DEBUG ON)  # 输出详细Boost 查找信息

set(TENSORRT_PATH "F:/TensorRT-10.8.0.43")
set(OpenCV_DIR "F:/opencv4_8_1/opencv/build")
set(OrbbecSDK_DIR "F:/OrbbecSDK2_11/lib")
set(PCL_DIR "F:/PCL 1.12.1/cmake")

set(AinstecSDK_PATH "F:/05.08ainstec-dualModel-obb/extension/CamSDK_CPP")

set(AinstecSDK_LIBRARIES "${AinstecSDK_PATH}/lib/win")
set(AinstecSDK_INCLUDE_DIRS "${AinstecSDK_PATH}/include")
set(AinstecSDK_3RDPARTY_INCLUDE_DIRS "${AinstecSDK_INCLUDE_DIRS}/3rdparty")
set(AinstecSDK_LIBTIFF_INCLUDE_DIRS 
    "${AinstecSDK_3RDPARTY_INCLUDE_DIRS}/libtiff/include")

file(GLOB AinstecSDK_MAIN_LIBRARIES "${AinstecSDK_LIBRARIES}/*.lib")
file(GLOB AinstecSDK_TIFF_LIBRARIES "${AinstecSDK_LIBRARIES}/libtiff/lib/*.lib")
# file(GLOB AinstecSDK_OPENCV_LIBRARIES "${AinstecSDK_LIBRARIES}/3rdparty/opencv/*.lib")
set(AinstecSDK_LIBS
    ${AinstecSDK_MAIN_LIBRARIES}
    ${AinstecSDK_TIFF_LIBRARIES}
    # ${AinstecSDK_OPENCV_LIBRARIES}
)

# Windows 特定配置
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
    add_definitions(
        -D_WIN32_WINNT=0x0601  # Windows 7 or later
        -DWIN32_LEAN_AND_MEAN
    )
endif()

find_package(Boost EXACT 1.78.0 REQUIRED COMPONENTS log
												    log_setup
												    thread
												    system
												    filesystem
												    date_time)
find_package(OpenCV EXACT 4.8.1 REQUIRED)
find_package(CUDA EXACT 11.8 REQUIRED)
find_package(OrbbecSDK REQUIRED)
find_package(PCL EXACT 1.12.1 REQUIRED)

set(SOURCES
	src/YOLO11.cpp
	# src/YoloDetector.cpp
	# src/AinstecCameraManager.cpp
	# src/OrbbecCameraManager.cpp
	# src/MainServer.cpp
	# src/ClientHandler.cpp
	src/preprocess.cu
	# src/Model.cpp
	# src/utils_c.c
	# src/utils_opencv.cpp
	# src/utils.cpp
	# src/DualModel.cpp
	src/AsyncLogger.cpp
	# src/ResultSaver.cpp
	# main.cpp
)

# set_source_files_properties(utils_c.c PROPERTIES LANGUAGE C)


add_executable(${PROJECT_NAME} ${SOURCES} main.cpp)
# add_library(${PROJECT_NAME} SHARED ${SOURCES})

set_target_properties(${PROJECT_NAME} PROPERTIES
    DEBUG_POSTFIX "d"
)
 target_compile_definitions(${PROJECT_NAME} 
     PUBLIC  
         VISION_EXPORTS
 )

target_include_directories(${PROJECT_NAME} PUBLIC
	${CMAKE_SOURCE_DIR}/include
	${OpenCV_INCLUDE_DIRS}
	${CUDA_INCLUDE_DIRS}
	${TENSORRT_PATH}/include
	${OrbbecSDK_DIR}/include
	${AinstecSDK_INCLUDE_DIRS}
    ${AinstecSDK_3RDPARTY_INCLUDE_DIRS}
    ${AinstecSDK_LIBTIFF_INCLUDE_DIRS}
	${Boost_INCLUDE_DIRS}
	${PCL_INCLUDE_DIRS}
)

set(cudnn_DIR ${CUDA_DIR})
set(cudnn_LIBS 
	"${cudnn_DIR}/lib/x64/cudnn.lib"
	"${cudnn_DIR}/lib/x64/cudnn64_8.lib"
	"${cudnn_DIR}/lib/x64/cudnn_adv_infer.lib"
	"${cudnn_DIR}/lib/x64/cudnn_adv_infer64_8.lib"
	"${cudnn_DIR}/lib/x64/cudnn_adv_train.lib"
	"${cudnn_DIR}/lib/x64/cudnn_adv_train64_8.lib"
	"${cudnn_DIR}/lib/x64/cudnn_cnn_train.lib"
	"${cudnn_DIR}/lib/x64/cudnn_cnn_train64_8.lib"
	"${cudnn_DIR}/lib/x64/cudnn_cnn_infer.lib"
	"${cudnn_DIR}/lib/x64/cudnn_cnn_infer64_8.lib"
	"${cudnn_DIR}/lib/x64/cudnn_ops_train.lib"
	"${cudnn_DIR}/lib/x64/cudnn_ops_train64_8.lib"
	"${cudnn_DIR}/lib/x64/cudnn_ops_infer.lib"
	"${cudnn_DIR}/lib/x64/cudnn_ops_infer64_8.lib"
)

set(TENSORRT_LIBS
	"${TENSORRT_PATH}/lib/nvinfer_10.lib"
	"${TENSORRT_PATH}/lib/nvonnxparser_10.lib"
	"${TENSORRT_PATH}/lib/nvinfer_dispatch_10.lib"
	"${TENSORRT_PATH}/lib/nvinfer_plugin_10.lib"
	"${TENSORRT_PATH}/lib/nvinfer_lean_10.lib"
	"${TENSORRT_PATH}/lib/nvinfer_vc_plugin_10.lib"
)

set(Boost_LIBRARIES
	Boost::log
	Boost::log_setup
	Boost::thread
	Boost::system
	Boost::filesystem
	Boost::date_time
)

target_link_libraries(${PROJECT_NAME} PUBLIC
	${OpenCV_LIBS}
	${CUDA_LIBRARIES}
	${TENSORRT_LIBS}
	${AinstecSDK_LIBS}
	# ob::OrbbecSDK
	${Boost_LIBRARIES}
	${cudnn_LIBS}
	# ws2_32
	${PCL_LIBRARIES}
)

option(SHOW_COST_DETAILS "Enable details in yolo11 time-costs" OFF)

if(SHOW_COST_DETAILS)
	add_definitions(-DSHOW_COST_DETAILS=1)
endif()

# message(STATUS "OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
# message(STATUS "CUDA_INCLUDE_DIRS=${CUDA_INCLUDE_DIRS}")
# message(STATUS "SHOW_COST_DETAILS=${SHOW_COST_DETAILS}")
# message(STATUS "Orbbec_INCLUDE_DIRS=${OrbbecSDK_INCLUDE_DIRS}")
# message(STATUS "Boost_INCLUDE_DIRS=${Boost_INCLUDE_DIRS}")
# message(STATUS "Boost_LIBRARIES=${Boost_LIBRARIES}")
message(STATUS "PCL_INCLUDE=${PCL_INCLUDE_DIRS}")