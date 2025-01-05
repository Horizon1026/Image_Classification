cmake_minimum_required( VERSION 3.0 )

project( slam_utility )

set( CMAKE_BUILD_TYPE "Debug" )
set( CMAKE_CXX_FLAGS "-std=c++17 -O3 -g -Wall -pthread" )
set( CMAKE_EXPORT_COMPILE_COMMANDS ON )
set( CMAKE_CXX_STANDARD 20 )

# Add all relative components of slam utility.
set( SLAM_UTILITY_PATH ${PROJECT_SOURCE_DIR}/../Slam_Utility )
if ( NOT TARGET lib_slam_utility_basic_type )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/basic_type ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_basic_type )
endif()
if ( NOT TARGET lib_slam_utility_log )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/log ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_log )
endif()
if ( NOT TARGET lib_slam_utility_math )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/math ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_math )
endif()
if ( NOT TARGET lib_slam_utility_memory )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/memory ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_memory )
endif()
if ( NOT TARGET lib_slam_utility_operate )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/operate ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_operate )
endif()
if ( NOT TARGET lib_slam_utility_tick_tock )
    add_subdirectory( ${SLAM_UTILITY_PATH}/src/tick_tock ${PROJECT_SOURCE_DIR}/build/lib_slam_utility_tick_tock )
endif()

# Add stack backward.
if ( NOT TARGET lib_stack_backward )
    add_subdirectory( ${SLAM_UTILITY_PATH}/3rd_libraries/backward ${PROJECT_SOURCE_DIR}/build/lib_stack_backward )
endif()

# Create executable target to test_libtorch_cnn_mnist.
set( IMAGE_CLASSIFICATION_PATH ${PROJECT_SOURCE_DIR}/../Image_Classification )
if ( NOT TARGET lib_mnist_cnn )
    add_subdirectory( ${IMAGE_CLASSIFICATION_PATH}/src/single_label_classification/mnist_cnn ${PROJECT_SOURCE_DIR}/build/lib_mnist_cnn )
endif()
add_executable( test_libtorch_cnn_mnist
    test/test_libtorch_cnn_mnist.cpp
)
target_link_libraries( test_libtorch_cnn_mnist
    lib_slam_utility_basic_type
    lib_slam_utility_log
    lib_slam_utility_math
    lib_slam_utility_memory
    lib_slam_utility_operate
    lib_slam_utility_tick_tock

    lib_stack_backward
    lib_mnist_cnn
)
